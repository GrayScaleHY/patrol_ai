
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from pathlib import Path

def load_predictor(num_classes=1, score_thresh=0.5):
    """
    加载maskrcnn模型。
    """
    # Load mask-rcnn
    cfg = get_cfg()
    cfg.merge_from_file(Path(__file__).parent / "mask_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = str(Path(__file__).parent / "model_final.pth")
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    predictor = DefaultPredictor(cfg)
    return predictor


def inference(predictor, img):
    """
    mask-rcnn的inference代码，返回轮廓坐标
    args:
        predictor: 加载的maskrcnn模型，使用load_predictor函数加载
        img: image data
    return:
        contours:轮廓坐标。结构为[array[x, 1, 2], ..]
        boxes: 包住轮廓的框。结构为array[[xmin, ymin, xmax, ymax], ..]
        (masks, classes): mask 和 对应的类别
    """
    # maskrcnn推理，输出mask结果, 为false和true组成的矩阵。
    outputs = predictor(img) # 包含pred_masks, pred_boox, scores, pred_classes
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

if __name__ == '__main__':
    import glob
    import os

    mask_rcnn_weight = "models/pointer/1/inference/model_final.pth"
    img_file = "models/pointer/1/test.jpg"

    

    predictor = load_predictor()
    # for img_file in glob.glob(os.path.join("/home/yh/meter_recognition/test/test/meter","*.jpg"))[-1]:
    img = cv2.imread(img_file)
    print(type(img[0,0,0]))
    contours, boxes, _ = inference(predictor, img)

    out_file = img_file[:-4] + "mrcnn.jpg"

    cv2.drawContours(img,contours,-1,(0,0,255),1)
    cv2.rectangle(img, (int(boxes[0][0]), int(boxes[0][1])), (int(boxes[0][2]), int(boxes[0][3])), (0, 0, 255), thickness=2)
    cv2.imwrite(out_file, img)
    cv2.imwrite(out_file, img)

        

