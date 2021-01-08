import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator

from fcos_core.layers import Scale
from fcos_core.layers import DFConv2d


class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS  #False
        self.centerness_on_reg = cfg.MODEL.FCOS.CENTERNESS_ON_REG #False
        self.use_dcn_in_tower = cfg.MODEL.FCOS.USE_DCN_IN_TOWER   #False

        cls_tower = []
        bbox_tower = []
        fix_tower = [] #fixme
        ratio_tower = [] #fixme

        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            if self.use_dcn_in_tower and \
                    i == cfg.MODEL.FCOS.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d

            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())

            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

            fix_tower.append(conv_func(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True)) #fixme
            fix_tower.append(nn.GroupNorm(32, in_channels))   #fixme
            fix_tower.append(nn.ReLU())                       #fixme

            ratio_tower.append(conv_func(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True))  #fixme
            ratio_tower.append(nn.GroupNorm(32, in_channels))  #fixme
            ratio_tower.append(nn.ReLU())                      #fixme

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.add_module('fix_tower', nn.Sequential(*fix_tower))# fixme
        self.add_module('ratio_tower', nn.Sequential(*ratio_tower))# fixme

        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )
        #fixme 添加
        self.fix_reg = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.ratio_reg = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,self.fix_tower,self.ratio_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness,
                        self.fix_reg,self.ratio_reg]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        #fixme 添加
        fix_reg=[]
        ratio_reg=[]

        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)
            fix_tower = self.fix_tower(feature)  #fixme 添加
            ratio_tower = self.ratio_tower(feature) # fixme 添加

            logits.append(self.cls_logits(cls_tower))

            if self.centerness_on_reg:          #fixme 打开是否加入到回归的分支
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred)
                if self.training:
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                bbox_reg.append(torch.exp(bbox_pred))

            #fixme 添加（是不是fix的值不应该乘以尺寸比例？？？）
            #fix_pred=self.scales[l](self.fix_reg(fix_tower))
            fix_pred = self.fix_reg(fix_tower) #fixme 这样改后结果更差
            if self.norm_reg_targets:
                if self.training:
                    fix_reg.append(F.sigmoid(fix_pred))
                else:
                    fix_reg.append(F.sigmoid(fix_pred))
            else:
                fix_reg.append(F.sigmoid(fix_pred))

            # if self.centerness_on_reg:
            #     ratio_reg.append(F.sigmoid(self.ratio_reg(box_tower)))
            # else:
            #     ratio_reg.append(F.sigmoid(self.ratio_reg(cls_tower)))
            ratio_reg.append(F.sigmoid(self.ratio_reg(ratio_tower)))

        return logits, bbox_reg, centerness,fix_reg,ratio_reg


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()

        head = FCOSHead(cfg, in_channels)

        box_selector_test = make_fcos_postprocessor(cfg)  #todo 修改
        loss_evaluator = make_fcos_loss_evaluator(cfg)   #todo 修改
        self.head = head
        self.box_selector_test = box_selector_test       #todo 修改
        self.loss_evaluator = loss_evaluator             #todo 修改
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        #todo 添加
        box_cls, box_regression, centerness,fix_reg,ratio_reg = self.head(features)
        locations = self.compute_locations(features)
        if self.training:
            return self._forward_train(   #todo 修改
                locations, box_cls, 
                box_regression, 
                centerness,fix_reg,ratio_reg, targets
            )
        else:
            return self._forward_test(   #todo 修改
                locations, box_cls, box_regression,
                centerness, fix_reg, ratio_reg, images.image_sizes
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness,fix_reg,ratio_reg,targets):
        loss_box_cls, loss_box_reg, loss_centerness,loss_fix_reg,loss_ratio = self.loss_evaluator( #todo 修改
            locations, box_cls, box_regression, centerness,fix_reg,ratio_reg, targets
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness,
            "loss_fix_reg":loss_fix_reg,
            "loss_ratio":loss_ratio
        }
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness,fix_reg,ratio_reg,image_sizes):
        boxes = self.box_selector_test( #todo 修改
            locations, box_cls, box_regression,
            centerness, fix_reg, ratio_reg, image_sizes
        )
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride/4,  #debug  生成较密的中心点位置 stride-->stride/4
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride/4,
            dtype=torch.float32, device=device
        )

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + (stride //8 ) #debug 对坐标进行移动
        return locations

def build_fcos(cfg, in_channels):
    return FCOSModule(cfg, in_channels)
