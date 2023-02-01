
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from lib_decode_model import decode_model
import sys
import os
# sys.path.append('./detectron2')

def load_maskrcnn_model(mask_rcnn_weight, num_classes=1, score_thresh=0.3,decode=False):
    """
    加载maskrcnn模型。
    """
    # Load mask-rcnn
    cfg = get_cfg()
    cfg.merge_from_file(
        "/data/PatrolAi/maskrcnn/mask_rcnn_R_101_FPN_3x.yaml")
    if decode:
        mask_rcnn_weight = decode_model(mask_rcnn_weight)[0]
        cfg.MODEL.WEIGHTS = mask_rcnn_weight
    else:
        cfg.MODEL.WEIGHTS = mask_rcnn_weight
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.DATASETS.TEST = ("meter", )
    # cfg.MODEL.DEVICE='cpu'
    maskrcnn_weights = DefaultPredictor(cfg)
    if decode:
        os.remove(mask_rcnn_weight)
    return maskrcnn_weights

def sel_boxes(boxes, include=0.9):
    if len(boxes) < 1:
        return []
    rm_index = []
    for i in range(len(boxes)):
        b1 = boxes[i]
        b1_square = abs(b1[2] - b1[0]) * abs(b1[3] - b1[1])
        for b2 in boxes:
            if (b1[2] - b1[0]) * (b1[3] - b1[1]) >= (b2[2] - b2[0]) * (b2[3] - b2[1]):
                continue

            # 计算区域交集的面积
            lu = np.maximum(b1[0:2], b2[0:2]) # 计算区域交集的左上与右下坐标
            rd = np.minimum(b1[2:], b2[2:])
            intersection = np.maximum(0.0, rd - lu)
            inter_square = intersection[0] * intersection[1]

            include_score = inter_square / b1_square
            if include_score > include:
                rm_index.append(i)
                continue
    return rm_index


def inference_maskrcnn(maskrcnn_weights, img, include=0.9):
    """
    mask-rcnn的inference代码，返回轮廓坐标
    args:
        maskrcnn_weights: 加载的maskrcnn模型，使用load_maskrcnn_model函数加载
        img: image data
    return:
        contours:轮廓坐标。结构为[array[x, 1, 2], ..]
        boxes: 包住轮廓的框。结构为array[[xmin, ymin, xmax, ymax], ..]
        (masks, classes): mask 和 对应的类别
    """
    # maskrcnn推理，输出mask结果, 为false和true组成的矩阵。
    outputs = maskrcnn_weights(img) # 包含pred_masks, pred_boox, scores, pred_classes
    instances = outputs["instances"]
    masks = instances.pred_masks.to('cpu').numpy() #提取masks
    boxes = instances.pred_boxes.tensor.to('cpu').numpy() #提取boxes
    classes = instances.pred_classes.to('cpu').numpy()
    scores = instances.scores.to('cpu').numpy()

    if include is not None:
        rm_index = sel_boxes(boxes, include)
        masks = np.array([masks[i] for i in range(len(masks)) if i not in rm_index], dtype=bool)
        boxes = np.array([boxes[i] for i in range(len(boxes)) if i not in rm_index], dtype=np.float32)
        classes = np.array([classes[i] for i in range(len(classes)) if i not in rm_index], dtype=np.int64)
        scores = np.array([scores[i] for i in range(len(scores)) if i not in rm_index], dtype=np.float32)

    # 将masks转成轮廓contours。
    contours = []
    for mask in masks:
        mask = np.where(mask == 1, 255, 0).astype(
            np.uint8)  # False, True转为0， 1

        ## 转成轮廓格式，contour由多个坐标点组成。
        contour, hierarchy = cv2.findContours(
            mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        ## 由于一个mask可能会拟合出多个contour，因此取点数最多的contour
        if len(contour) < 1:
            continue
        contour_shape0 = [c.shape[0] for c in contour]
        contour = [contour[contour_shape0.index(max(contour_shape0))]]

        for c_ in contour:
            contours.append(c_)

    # cv2.drawContours(img,contours,-1,(0,0,255),1)
    # cv2.imwrite("/home/yh/meter_recognition/test/point_two_0_contours.jpg", img)
    return contours, boxes, (masks, classes, scores)

if __name__ == '__main__':
    line = [801, 302, 1352, 643] # [x1, y1, x2, y2]
    arc = [1058, 462, 1327, 375, 1250, 732]# [xc, yc, x1, y1, x2, y2]

        

