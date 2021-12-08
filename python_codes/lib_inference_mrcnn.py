
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import sys
import math
from lib_analysis_meter import segment2angle
sys.path.append('./detectron2')

def intersection_arc(line, arc):
    """
    计算射线line与圆弧arc的交点坐标，最多返回一个坐标值。
    args:
        line: [x1, y1, x2, y2]
        arc: [xc, yc, x1, y1, x2, y2], 注意:半径r取(x1,y1)到圆心的距离。
    return:
        (x1, y1): 交点坐标。也可能是None，表示直线与弧线无交点。
    """
    ## line = [130, 296, 216, 328] # [x1, y1, x2, y2]
    ## arc = [398, 417, 116, 413, 229, 174] # [xc, yc, x1, y1, x2, y2]

    line = np.array(line, dtype=float) ## int转float
    arc = np.array(arc, dtype=float)
    # 根据线段到圆心的远近，确定坐标的先后
    if ((line[0]-arc[0])**2 + (line[1]-arc[1])**2) > ((line[2]-arc[0])**2 + (line[3]-arc[1])**2):
        line = [line[2], line[3], line[0], line[1]]

    ## 定义直线和圆形方程式
    lx1 = line[0]; ly1 = line[1]; lx2 = line[2]; ly2 = line[3]
    xc = arc[0]; yc = arc[1]
    ax1 = arc[2]; ay1 = arc[3]; ax2 = arc[4]; ay2 = arc[5]
    # 直线: y = a * x + b
    if lx1 == lx2:
        a = None
    else:
        a = (ly1 - ly2) / (lx1 - lx2)
        b = (ly1 - a * lx1)
    # 圆形: (x - xc)^2 + (y - yc)^2 = r2
    r2 = (yc - ay1) ** 2 + (xc - ax1) ** 2

    ## 线段至少一个点在圆内。
    if ((lx1 - xc) ** 2 + (ly1 - yc) ** 2) >= r2:
        return None

    ## 计算直线与圆弧的交点坐标
    if a is None:
        ## 直线斜率不存在，线段垂直与x轴
        x1 = lx1
        x2 = lx1
        y1 = math.sqrt(r2 - (x1 - xc) ** 2) - yc
        y2 = -math.sqrt(r2 - (x1 - xc) ** 2) - yc
    else:
        ## 有斜率的情况下，联立直线和圆形方程组得: Ax^2 + Bx + C = 0
        A = 1 + a ** 2
        B = 2 * a * (b - yc) - 2 * xc
        C = xc ** 2 + (b - yc) ** 2 - r2
        detl = B ** 2 - 4 * A * C
        if detl <= 0: # 相切或不相交
            return None
        else:
            ## 二元一次方程组求解公式
            x1 = (-B - math.sqrt(detl)) / (2 * A)
            x2 = (-B + math.sqrt(detl)) / (2 * A)
            y1 = a * x1 + b
            y2 = a * x2 + b
    
    ## 取离线段末端最近的交点坐标作为唯一的交点坐标。
    if ((lx2-x1)**2+(ly2-y1)**2) <= ((lx2-x2)**2+(ly2-y2)**2):
        x = x1; y = y1
    else:
        x = x2; y = y2

    ## 判断交点坐标是否在圆弧内
    ang1 = segment2angle((xc, yc), (ax1, ay1))
    ang2 = segment2angle((xc, yc), (x, y))
    ang3 = segment2angle((xc, yc), (ax2, ay2))
    if ang2 <= ang1:
        ang2_ = ang1 - ang2
    else:
        ang2_ = ang1 + (360 - ang2) # 计算圆弧arc的夹角度数
    if ang3 <= ang1:
        ang3_ = ang1 - ang3
    else:
        ang3_ = ang1 + (360 - ang3) # 计算圆弧arc的夹角度数
    if ang2_ > ang3_:
        return None
    
    return int(x), int(y)


def intersection_segment(line, segment):
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


def load_maskrcnn_model(mask_rcnn_weight, num_classes=1, score_thresh=0.5):
    """
    加载maskrcnn模型。
    """
    # Load mask-rcnn
    cfg = get_cfg()
    cfg.merge_from_file(
        "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = mask_rcnn_weight
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
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
        (masks, classes): mask 和 对应的类别
    """
    # maskrcnn推理，输出mask结果, 为false和true组成的矩阵。
    outputs = maskrcnn_weights(img) # 包含pred_masks, pred_boox, scores, pred_classes
    instances = outputs["instances"]
    masks = instances.pred_masks.to('cpu').numpy() #提取masks
    boxes = instances.pred_boxes.tensor.to('cpu').numpy() #提取boxes
    classes = instances.pred_classes.to('cpu').numpy()

    # 将masks转成轮廓contours。
    contours = []
    for mask in masks:
        mask = np.where(mask == 1, 255, 0).astype(
            np.uint8)  # False, True转为0， 1

        ## 转成轮廓格式，contour由多个坐标点组成。
        contour, hierarchy = cv2.findContours(
            mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        ## 由于一个mask可能会拟合出多个contour，因此取点数最多的contour
        contour_shape0 = [c.shape[0] for c in contour]
        contour = [contour[contour_shape0.index(max(contour_shape0))]]

        for c_ in contour:
            contours.append(c_)

    # cv2.drawContours(img,contours,-1,(0,0,255),1)
    # cv2.imwrite("/home/yh/meter_recognition/test/point_two_0_contours.jpg", img)
    return contours, boxes, (masks, classes)


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
            coor = intersection_segment(line, seg) # 求直线与线段的交点坐标
            if coor is not None and coor not in coors:
                coors.append(coor)
                segment = segment + list(coor)
        
        assert len(segment) == 4, str(segment) + " is wrong!"
        segments.append(segment)

    return segments


if __name__ == '__main__':
    import glob
    import os

    mask_rcnn_weight = '/data/inspection/maskrcnn/pointer.pth'
    img_file = "/home/yh/app_inspection/python_codes/recognition_resutl/08-30-16-08-37/raw_img_meter_rec.jpg"

    maskrcnn_weights = load_maskrcnn_model(mask_rcnn_weight)
    # for img_file in glob.glob(os.path.join("/home/yh/meter_recognition/test/test/meter","*.jpg"))[-1]:
    img = cv2.imread(img_file)
    contours, boxes, _ = inference_maskrcnn(maskrcnn_weights, img)
    segments = contour2segment(contours, boxes)
    print(segments)

    for segment in segments:
        cv2.line(img, (segment[0], segment[1]),
                (segment[2], segment[3]), (0, 255, 0), 2)  # 图像上画直线

    out_file = img_file[:-4] + "mrcnn.jpg"

    # cv2.drawContours(img,contours,-1,(0,0,255),1)
    # cv2.rectangle(img, (int(boxes[0][0]), int(boxes[0][1])), (int(boxes[0][2]), int(boxes[0][3])), (0, 0, 255), thickness=2)
    cv2.imwrite(out_file, img)
    # cv2.imwrite(out_file, img)

        

