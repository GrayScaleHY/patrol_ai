
import sys
sys.path.append('./detectron2')
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import numpy as np
import cv2


def calc_abc_from_line_2d(x0, y0, x1, y1):
    """
    与lines_intersection配合使用
    """
    a = y0 - y1
    b = x1 - x0
    c = x0*y1 - x1*y0
    return a, b, c


def lines_intersection(segment1, segment2):
    """
    计算两直线交点坐标
    args:
        segment: [x0, y0, x1, y1]
    """
    # x1y1x2y2
    a0, b0, c0 = calc_abc_from_line_2d(*segment1)
    a1, b1, c1 = calc_abc_from_line_2d(*segment2)
    D = a0 * b1 - a1 * b0
    if D == 0:
        return None
    x = (b0 * c1 - b1 * c0) / D
    y = (a1 * c0 - a0 * c1) / D
    return x, y


def load_maskrcnn_model(mask_rcnn_weight):
    """
    加载maskrcnn模型。
    """
    # Load mask-rcnn
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = mask_rcnn_weight
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.DATASETS.TEST = ("meter", )
    maskrcnn_weights = DefaultPredictor(cfg)
    return maskrcnn_weights


def inference_maskrcnn(maskrcnn_weights, img):
    """
    mask-rcnn的inference代码，返回轮廓坐标
    args:
        maskrcnn_weights: 加载的maskrcnn模型，使用load_maskrcnn_model函数加载
        img: image data
    return:
        contours:轮廓坐标。结构为[array[x, 1, 2], ..]
    """
    ## maskrcnn推理，输出mask结果, 为false和true组成的矩阵。
    outputs = maskrcnn_weights(img)  # outputs结构：{'instances': tensor}
    masks = np.asarray(outputs["instances"].to("cpu").pred_masks) ## tensor 转numpy

    ## 将masks转成轮廓contours。
    contours = []
    for mask in masks:
        mask = np.where(mask == 1, 255, 0).astype(np.uint8) # False, True转为0， 1

        ## 转成轮廓格式，contour由多个坐标点组成。
        contour, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for c_ in contour:
            contours.append(c_)
    
    # cv2.drawContours(img,contours,-1,(0,0,255),1)
    # cv2.imwrite(img_file[:-4]+"_contours.jpg", img)
    return contours


def contour2segment(contours, img):
    """
    将轮廓拟合成线段。
    args:
        contours: 轮廓的坐标集, [array[x, 1, 2], ..]
        img: image data
    return:
        segments: 线段，结构：[[x0, y0, x1, y1], ..]
    """
    segments = []
    for contour in contours:

        ## 直接拟合直线，返回[cosΘ, sinΘ, x0, y0]
        fit_line = cv2.fitLine(contour, cv2.DIST_HUBER, 0, 0.01, 0.01)
        fit_line = np.squeeze(fit_line)  # 去除冗余的维度

        ## 拟合直线的性质
        cos_l = fit_line[0]
        sin_l = fit_line[1]
        x_l = fit_line[2]
        y_l = fit_line[3]
        tan_l = sin_l / cos_l

        ## 求出当y=0或y=max时直线上的坐标，即直线与图像上下两边交点坐标。返回线段。
        parm0 = (0 - y_l) / sin_l
        parm1 = (img.shape[0] - y_l) / sin_l
        coor_min = tuple(np.array([x_l, y_l]) + (parm0 * np.array([cos_l, sin_l])))
        coor_max = tuple(np.array([x_l, y_l]) + (parm1 * np.array([cos_l, sin_l])))
        segment_l = [coor_min[0], coor_min[1], coor_max[0], coor_max[1]]

        ## 求能框住轮廓的最小矩形框的四个顶点坐标。
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.int64)  # 获取矩形四个定点坐标

        ## 获取矩形框上的线段
        if tan_l > 0:
            segment_0 = [box[0][0],box[0][1],box[1][0],box[1][1]]
            segment_1 = [box[2][0],box[2][1],box[3][0],box[3][1]]
        else:
            segment_0 = [box[0][0],box[0][1],box[3][0],box[3][1]]
            segment_1 = [box[2][0],box[2][1],box[1][0],box[1][1]]

        ## 求矩形框上线段与直线的交点坐标
        coor_0 = lines_intersection(segment_0,segment_l)
        coor_1 = lines_intersection(segment_1,segment_l)

        segment = [int(coor_0[0]), int(coor_0[1]), int(coor_1[0]), int(coor_1[1])]

        segments.append(segment)

    return segments


if __name__ == '__main__':

    mask_rcnn_weight = '/home/yh/meter_recognition/detectron2/run/model_final.pth'
    img_file = "/home/yh/meter_recognition/test/1_155614_0_meter.jpg"

    maskrcnn_weights = load_maskrcnn_model(mask_rcnn_weight)
    img = cv2.imread(img_file)
    contours = inference_maskrcnn(maskrcnn_weights, img)
    segments = contour2segment(contours, img)
    print(segments)
    
    for segment in segments:    
        cv2.line(img, (segment[0], segment[1]), (segment[2], segment[3]), (0, 255, 0), 2)  # 图像上画直线

    out_file = img_file[:-4] + "mrcnn.jpg"
    cv2.imwrite(out_file, img)
