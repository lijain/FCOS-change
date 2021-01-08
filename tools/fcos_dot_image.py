"""
20201020: 根据fcos的中心点将反解后的中心点画上
"""
import math
import cv2 as cv
import os
import numpy as np


def draw(inputimage, outputimage):
    locations_ori=dot(8, 8)
    #locations_ma = dot(16, 8)

    img=cv.imread(inputimage)
    for num in range(locations_ori.shape[0]):
        x=locations_ori[num][0]
        y=locations_ori[num][1]
        cv.circle(img, (x, y), 1, (0, 0, 255), 1)

    # for num in range(locations_ma.shape[0]):
    #     x1 = locations_ma[num][0]
    #     y1 = locations_ma[num][1]
    #     cv.circle(img, (x1, y1), 1, (0, 255, 0), 1)
    cv.imwrite(outputimage, img)

def dot(stride,stride_step):
    w = 128
    h = 128

    shifts_x = np.arange(0, w * stride, step=stride_step)
    shifts_y = np.arange(0, h * stride, step=stride_step)

    shift_y, shift_x = np.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = np.stack((shift_x, shift_y), axis=1) + (stride_step // 2)

    return locations

if __name__=="__main__":
    save_path = "/data/maq/DataSet/DOTA_train/dota_20200501/trainval_cut/fcos_visual_imagedota"
    images_path = "/data/maq/DataSet/DOTA_train/dota_20200501/trainval_cut/train_dota_nonull/images"

    tempfilename = os.listdir(images_path)

    for file in tempfilename:
        inputimage=os.path.join(images_path,file)
        outputimage=os.path.join(save_path,file)

        draw(inputimage, outputimage)
    print("finished!!!")
