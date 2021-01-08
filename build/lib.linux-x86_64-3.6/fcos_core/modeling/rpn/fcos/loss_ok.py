"""
This file contains specific functions for computing losses of FCOS
file
"""

import torch
from torch.nn import functional as F
from torch import nn
import os
from ..utils import concat_box_prediction_layers
from fcos_core.layers import IOULoss
from fcos_core.layers import SigmoidFocalLoss
from fcos_core.layers import smooth_l1_loss   #todo 添加
from fcos_core.modeling.matcher import Matcher
from fcos_core.modeling.utils import cat
from fcos_core.structures.boxlist_ops import boxlist_iou
from fcos_core.structures.boxlist_ops import cat_boxlist
from fcos_core.modeling.box_coder import BoxCoder,RatioCoder,FixCoder

INF = 100000000
def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,
            cfg.MODEL.FCOS.LOSS_ALPHA
        )
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.center_sampling_radius = cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS  #false

        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        #todo 添加
        self.fixcoder =FixCoder()
        self.ratiocoder=RatioCoder()

    #判断在标注框中的点
    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1.0):
        '''
        This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
        '''
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(
                xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
            )
            center_gt[beg:end, :, 1] = torch.where(
                ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
            )
            center_gt[beg:end, :, 2] = torch.where(
                xmax > gt[beg:end, :, 2],
                gt[beg:end, :, 2], xmax
            )
            center_gt[beg:end, :, 3] = torch.where(
                ymax > gt[beg:end, :, 3],
                gt[beg:end, :, 3], ymax
            )
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0

        return inside_gt_bbox_mask

    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )
        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points] #每个level的点数
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)    #包含所有的采样点
        #fixme 为其添加过滤后的fix_targets,ratio_targets
        labels, reg_targets,fix_targets,ratio_targets= self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)
            #fixme 添加fix_targets,ratio_targets
            fix_targets[i]=torch.split(fix_targets[i],num_points_per_level,dim=0)
            ratio_targets[i]=torch.split(ratio_targets[i],num_points_per_level,dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        #fixme 添加
        fix_targets_level_first=[]
        ratio_targets_level_first=[]

        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )

            reg_targets_per_level = torch.cat([
                reg_targets_per_im[level]
                for reg_targets_per_im in reg_targets
            ], dim=0)
            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]   #[8, 16, 32, 64, 128]
            reg_targets_level_first.append(reg_targets_per_level)

            #fixme 添加
            ratio_targets_level_first.append(
                torch.cat([ratio_per_im[level] for ratio_per_im in ratio_targets],dim=0)
            )

            fix_targets_per_level=torch.cat([
                fix_targets_per_im[level]
                for fix_targets_per_im in fix_targets
            ],dim=0)
            fix_targets_level_first.append(fix_targets_per_level)
        # print("###labels_level_first:", len(labels_level_first),labels_level_first[0].shape, type(labels_level_first[0]))
        # print("###ratio_targets_level_first:",len(ratio_targets_level_first), ratio_targets_level_first[0].shape, type(ratio_targets_level_first[0]))
        # print("###reg_targets_level_first:", len(reg_targets_level_first),reg_targets_level_first[0].shape, type(reg_targets_level_first[0]))
        # print("###fix_targets_level_first:", len(fix_targets_level_first),fix_targets_level_first[0].shape, type(fix_targets_level_first[0]))
        # os._exit()
        return labels_level_first, reg_targets_level_first,fix_targets_level_first,ratio_targets_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        #fixme 添加
        fix_targets=[]
        ratio_targets=[]

        xs, ys = locations[:, 0], locations[:, 1]
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            #assert targets_per_im.mode == "xyxy"  #fixme 屏蔽
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            area = targets_per_im.area()
            #print("####bboxes:",bboxes,bboxes.shape)
            #fixme 添加 计算出rbox
            device = bboxes.device
            rbox_mask = targets_per_im.get_field("masks")
            # polygons = list(map(lambda x: x.polygons[0], rbox_mask.instances.polygons))
            # rbox = torch.stack(polygons, axis=0)
            # rbox_targets_per_im=rbox.expand(xs.size(0),rbox.size(0),rbox.size(1))
            rfix = self.fixcoder.encode(rbox_mask.instances.polygons) #todo 这样写可以避免后续对Gpu使用不充分
            rratio= self.ratiocoder.encode(rbox_mask.instances.polygons)
            # print("####loss_fix_rbox:", rfix, rfix.shape)
            # print("####loss_rratio_rbox:", rratio, rratio.shape)
            # os._exit()
            fix_targets_per_im = rfix.expand(xs.size(0), rfix.size(0), rfix.size(1)).cuda(device)
            ratio_targets_per_im=rratio.expand(xs.size(0), rratio.size(0), rratio.size(1)).cuda(device)
            #ratio_targets_per_im = rratio.squeeze().cuda(device) #fixme 尝试改正

            l = xs[:, None] - bboxes[:, 0][None]  #计算target中的[l,t,r,b]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
            # print("###labels_per_im:",labels_per_im.shape,type(labels_per_im))
            # print("###ratio_targets_per_im:", ratio_targets_per_im.shape,type(ratio_targets_per_im))
            # print("###reg_targets_per_im:", reg_targets_per_im.shape,type(reg_targets_per_im))
            # print("###fix_targets_per_im:", fix_targets_per_im.shape,type(fix_targets_per_im))
            #so._exit()
            # #torch.Size([9]) <class 'torch.Tensor'>
            # #torch.Size([9]) <class 'torch.Tensor'>
            # #torch.Size([21824, 9, 4]) <class 'torch.Tensor'>
            # # torch.Size([21824, 9, 4]) <class 'torch.Tensor'>
            #todo  删除中心不在标注框内的点
            if self.center_sampling_radius > 0:
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    self.fpn_strides,
                    self.num_points_per_level,
                    xs, ys,
                    radius=self.center_sampling_radius
                )
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0
            #todo 类似FPN结构删除最大不在某一层的框
            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            #todo 如果标注框还在重复，按照之前的策略选择面积最小的框
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            reg_targets.append(reg_targets_per_im)
            labels_per_im = labels_per_im[locations_to_gt_inds]
            #print("###labels_per_im0:", labels_per_im.shape, type(labels_per_im))
            labels_per_im[locations_to_min_area == INF] = 0
            #print("###labels_per_im1:", labels_per_im.shape, type(labels_per_im))
            labels.append(labels_per_im)
            #print("###labels_per_im2:", labels_per_im.shape, type(labels_per_im))

            # fixme 添加斜框的坐标点
            fix_targets_per_im=fix_targets_per_im[range(len(locations)), locations_to_gt_inds]
            ratio_targets_per_im=ratio_targets_per_im[range(len(locations)), locations_to_gt_inds]  #fixme ??
            #print("###ratio_targets_per_im0:", ratio_targets_per_im.shape, type(ratio_targets_per_im))
            fix_targets.append(fix_targets_per_im)
            ratio_targets.append(ratio_targets_per_im)
            # print("###labels_per_im:",labels_per_im.shape,type(labels_per_im))
            # print("###ratio_targets_per_im:", ratio_targets_per_im.shape, type(ratio_targets_per_im))
            # print("###reg_targets_per_im:", reg_targets_per_im.shape, type(reg_targets_per_im))
            # print("###fix_targets_per_im:", fix_targets_per_im.shape, type(fix_targets_per_im))
            # os._exit()
            #torch.Size([21824]) <class 'torch.Tensor'>
            #torch.Size([21824]) <class 'torch.Tensor'>
            #torch.Size([21824, 4]) <class 'torch.Tensor'>
            #torch.Size([21824, 4]) <class 'torch.Tensor'>
        return labels, reg_targets,fix_targets,ratio_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression,centerness,fix_regression,ratio_regression,targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            fix_regression (list[Tensor])    #todo 添加
            ratio_regression(list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
            fix_loss  (Tensor)   #todo 添加
            ratio_loss (Tensor)
        """
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        #fixme 添加target的fix_targets，ratio_targets的准备数据
        labels, reg_targets,fix_targets,ratio_targets = self.prepare_targets(locations, targets)

        #todo 预测的值
        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        #fixme
        fix_regession_flatten = []
        ratio_regession_flatten = []

        #todo targets的设置
        labels_flatten = []
        reg_targets_flatten = []
        #fixme 添加
        fix_targets_flatten =[]
        ratio_targets_flatten = []

        for l in range(len(labels)):
            #todo 预测的值
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            centerness_flatten.append(centerness[l].reshape(-1))
            #fixme 添加
            fix_regession_flatten.append(fix_regression[l].permute(0,2,3,1).reshape(-1,4))
            #ratio_regession_flatten.append(ratio_regression[l].permute(0,2,3,1).reshape(-1))
            #fixme 尝试改正
            ratio_regession_flatten.append(ratio_regression[l].reshape(-1))

            #todo target的设置
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            #fixme 添加
            fix_targets_flatten.append(fix_targets[l].reshape(-1,4))
            ratio_targets_flatten.append(ratio_targets[l].reshape(-1))

            # print("###box_cls[l]:", box_cls[l].shape, type(box_cls[l]))
            # print("###ratio_regression[l]:", ratio_regression[l].shape, type(ratio_regression[l]))
            #
            # print("###box_cls_flatten:",box_cls_flatten[0].shape,type(box_cls_flatten[0]))
            # print("###ratio_regession_flatten:", ratio_regession_flatten[0].shape, type(ratio_regession_flatten[0]))
            # print("###centerness_flatten:", centerness_flatten[0].shape, type(centerness_flatten[0]))
            # print("###box_regression_flatten:", box_regression_flatten[0].shape, type(box_regression_flatten[0]))
            # print("###fix_regession_flatten:", fix_regession_flatten[0].shape, type(fix_regession_flatten[0]))
            #
            # print("###labels_flatten:", labels_flatten[0].shape, type(labels_flatten[0]))
            # print("###ratio_targets_flatten:", ratio_targets_flatten[0].shape, type(ratio_targets_flatten[0]))
            # print("###reg_targets_flatten:", reg_targets_flatten[0].shape, type(reg_targets_flatten[0]))
            # print("###fix_targets_flatten:", fix_targets_flatten[0].shape, type(fix_targets_flatten[0]))
            # os._exit()

        #todo 预测的值
        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        #fixme 添加
        fix_regession_flatten = torch.cat(fix_regession_flatten,dim=0)
        ratio_regession_flatten = torch.cat(ratio_regession_flatten,dim=0)

        #todo target的设置
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)
        #fixme 添加
        fix_targets_flatten = torch.cat(fix_targets_flatten, dim=0)
        ratio_targets_flatten = torch.cat(ratio_targets_flatten,dim=0)

        #todo 选择正例坐标   ################
        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        # todo 预测的值
        box_regression_flatten = box_regression_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]
        # fixme 添加
        fix_regession_flatten = fix_regession_flatten[pos_inds]
        ratio_regession_flatten = ratio_regession_flatten[pos_inds]

        # todo target的设置
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        #fixme 添加
        fix_targets_flatten = fix_targets_flatten[pos_inds]
        ratio_targets_flatten = ratio_targets_flatten[pos_inds]

        num_gpus = get_num_gpus()
        # sync num_pos from all gpus
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / num_pos_avg_per_gpu

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)

            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss
            sum_centerness_targets_avg_per_gpu = \
                reduce_sum(centerness_targets.sum()).item() / float(num_gpus)

            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,       #pred
                reg_targets_flatten,          #target
                centerness_targets
            ) / sum_centerness_targets_avg_per_gpu

            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            ) / num_pos_avg_per_gpu

            #fixme 添加fix与ratio的损失
            fix_loss=smooth_l1_loss(
                fix_regession_flatten,
                fix_targets_flatten,
                size_average=False,
                beta=1. / 3.,
            ) / num_pos_avg_per_gpu

            ratio_loss=smooth_l1_loss(
                ratio_regession_flatten,
                ratio_targets_flatten,
                size_average = False,
                beta = 1. /3,
            ) / num_pos_avg_per_gpu

            ratio_loss = ratio_loss * 4
            # print("###num_pos_avg_per_gpu:",num_pos_avg_per_gpu)
            # print("###sum_centerness_targets_avg_per_gpu:", sum_centerness_targets_avg_per_gpu)
            # print("###centerness_targets:", centerness_targets[:2],centerness_targets.shape)
        else:
            reg_loss = box_regression_flatten.sum()
            reduce_sum(centerness_flatten.new_tensor([0.0]))
            centerness_loss = centerness_flatten.sum()
            #todo 添加
            fix_loss = fix_regession_flatten.sum()
            ratio_loss = ratio_regession_flatten.sum()
        # print("###cls_loss:", cls_loss)
        # print("###reg_loss:", reg_loss)
        # print("###centerness_loss:", centerness_loss)
        # print("###fix_loss:", fix_loss)
        # print("###ratio_loss:", ratio_loss)
        #os._exit()
        return cls_loss, reg_loss, centerness_loss, fix_loss, ratio_loss


def make_fcos_loss_evaluator(cfg):

    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator
