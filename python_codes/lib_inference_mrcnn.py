
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import sys
sys.path.append('./detectron2')

def intersection_point(line, segment):
    """
    求直线穿过线段的交点坐标。
    agrs:
        line: 在直线上的两点坐标[x0, y0, x1, y1]
        segment: 线段两端点的坐标[x0, y0, x1, y1]
    return:
        (x, y): 交点坐标（如果没交点，返回None）
    """
    ## 如果线段或直线只包含一个坐标点，则报错
    assert (line[0], line[1]) != (line[2], line[3]
                                  ), "line needs at least two different points"
    assert (segment[0], segment[1]) != (segment[2], segment[3]
                                        ), "segment needs at least two different points"

    ## 将坐标值转为浮点型，方便计算
    line = np.array(line, dtype=float)
    segment = np.array(segment, dtype=float)
    
    ## 计算直线的斜率k_l和偏执值b_l
    if line[2] !=  line[0]:
        k_l = (line[3] - line[1]) / (line[2] - line[0]) # k = (y2 - y1) / (x2 - x1)
        b_l = line[1] - line[0] * k_l  # b = y1 - x1 * k
    else:
        k_l = None
    ## 计算线段的斜率k_s和偏执值b_s
    if segment[2] !=  segment[0]: # 判断直线是否有斜率
        k_s = (segment[3] - segment[1]) / (segment[2] - segment[0]) # k = (y2 - y1) / (x2 - x1)
        b_s = segment[1] - segment[0] * k_s  # b = y1 - x1 * k
    else:
        k_s = None

    if k_l == k_s: ## 如果直线和线段的斜率相同，则返回None
        return None

    if k_l is not None:
        ## 根据线段两端点是否在直线两边，判断直线与线段是否有交点。
        ds_y1 = segment[1] - (k_l * segment[0] + b_l)
        ds_y2 = segment[3] - (k_l * segment[2] + b_l)
        if ds_y1 * ds_y2 > 0:
            return None
        
        if k_s is None:
            x = segment[0]
        else:
            x = (b_s - b_l) / (k_l - k_s) #联立方程式计算x
        y = k_l * x + b_l
    
    else:
        ##根据线段两端点是否再直线两边，判断直线与线段是否有交点。
        if (segment[0] - line[0]) * (segment[2] - line[0]) > 0:
            return None
        
        x = line[0]
        y = k_s * x + b_s
    
    return int(x), int(y)


def load_maskrcnn_model(mask_rcnn_weight):
    """
    加载maskrcnn模型。
    """
    # Load mask-rcnn
    cfg = get_cfg()
    cfg.merge_from_file(
        "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
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
        boxes: 包住轮廓的框。结构为array[[xmin, ymin, xmax, ymax], ..]
    """
    # maskrcnn推理，输出mask结果, 为false和true组成的矩阵。
    outputs = maskrcnn_weights(img) # 包含pred_masks, pred_boox, scores, pred_classes
    instances = outputs["instances"]
    masks = instances.pred_masks.to('cpu').numpy() #提取masks
    boxes = instances.pred_boxes.tensor.to('cpu').numpy() #提取boxes

    # 将masks转成轮廓contours。
    contours = []
    for mask in masks:
        mask = np.where(mask == 1, 255, 0).astype(
            np.uint8)  # False, True转为0， 1

        # 转成轮廓格式，contour由多个坐标点组成。
        contour, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for c_ in contour:
            contours.append(c_)

    # cv2.drawContours(img,contours,-1,(0,0,255),1)
    # cv2.imwrite(img_file[:-4]+"_contours.jpg", img)
    return contours, boxes


def contour2segment(contours, boxes):
    """
    将轮廓拟合成线段。
    args:
        contours: 轮廓的坐标集, [array[x, 1, 2], ..]
        img: image data
    return:
        segments: 线段，结构：[[x0, y0, x1, y1], ..]
    """
    assert len(contours) == len(boxes), "contours not match boxes!"
    segments = []
    for i, contour in enumerate(contours):
        ## 轮廓拟合成直线，返回[cosΘ, sinΘ, x0, y0]
        fit_line = cv2.fitLine(contour, cv2.DIST_HUBER, 0, 0.01, 0.01)
        fit_line = np.squeeze(fit_line)  # 去除冗余的维度
        cos_l = fit_line[0]
        sin_l = fit_line[1]
        x_l = fit_line[2]
        y_l = fit_line[3]

        ## 求能框住轮廓的最小矩形框的四个顶点坐标。
        # rect = cv2.minAreaRect(contour)
        # box = cv2.boxPoints(rect).astype(np.int64)  # 获取矩形四个定点坐标

        ## 将拟合的直线转换成[x1, y1, x2, y2]的形式
        if cos_l != 0: #是否有斜率
            tan_l = sin_l / cos_l
            line = [x_l, y_l, x_l + 1, y_l + tan_l * 1]
        else:
            line = [x_l, y_l, x_l, y_l + 1]

        ## 将box写成四条线段的形式
        box = list(boxes[i])
        seg_list = [[box[0], box[1], box[2], box[1]],
                    [box[2], box[1], box[2], box[3]],
                    [box[2], box[3], box[0], box[3]],
                    [box[0], box[3],box[0], box[1]]]
        
        ## 求直线与矩形框的交点坐标，理论上会有两个交点。
        coors = []
        segment = []
        for seg in seg_list:
            coor = intersection_point(line, seg) # 求直线与线段的交点坐标
            if coor is not None and coor not in coors:
                coors.append(coor)
                segment = segment + list(coor)
        
        assert len(segment) == 4, str(segment) + " is wrong!"
        segments.append(segment)

    return segments


if __name__ == '__main__':

    mask_rcnn_weight = '/home/yh/meter_recognition/detectron2/run/model_final.pth'
    img_file = "/home/yh/meter_recognition/test/#0294_org.jpg"

    maskrcnn_weights = load_maskrcnn_model(mask_rcnn_weight)
    img = cv2.imread(img_file)
    contours, boxes = inference_maskrcnn(maskrcnn_weights, img)
    segments = contour2segment(contours, boxes, img)
    print(segments)

    for segment in segments:
        cv2.line(img, (segment[0], segment[1]),
                 (segment[2], segment[3]), (0, 255, 0), 2)  # 图像上画直线

    out_file = img_file[:-4] + "mrcnn.jpg"

    # cv2.drawContours(img,contours,-1,(0,0,255),1)
    # cv2.rectangle(img, (int(boxes[0][0]), int(boxes[0][1])), (int(boxes[0][2]), int(boxes[0][3])), (0, 0, 255), thickness=2)
    cv2.imwrite(out_file, img)
    # cv2.imwrite(out_file, img)

