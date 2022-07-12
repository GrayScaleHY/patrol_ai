
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from pathlib import Path
from inference.centernet.config import add_centernet_config

def load_predictor(score_thresh=0.5):
    """
    加载detectron2模型。
    """
    # Load mask-rcnn
    cfg = get_cfg()
    add_centernet_config(cfg)
    cfg.merge_from_file(Path(__file__).parent / "Base-CenterNet2.yaml")
    cfg.MODEL.WEIGHTS = str(Path(__file__).parent / "model_final.pth")
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = score_thresh
    predictor = DefaultPredictor(cfg)
    return predictor


def inference(predictor, img):
    """
    detectron2的inference代码，返回轮廓坐标
    args:
        predictor: 加载的detectron2模型，使用load_maskrcnn_model函数加载
        img: image data
    return:
        contours:轮廓坐标。结构为[array[x, 1, 2], ..]
        boxes: 包住轮廓的框。结构为array[[xmin, ymin, xmax, ymax], ..]
        (masks, classes): mask 和 对应的类别
    """
    # maskrcnn推理，输出mask结果, 为false和true组成的矩阵。
    outputs = predictor(img) # 包含pred_masks, pred_boox, scores
    instances = outputs["instances"]
    scores = instances.scores.to('cpu').numpy()
    boxes = instances.pred_boxes.tensor.to('cpu').numpy() #提取boxes
    classes = instances.pred_classes.to('cpu').numpy()

    return boxes, scores, classes

if __name__ == '__main__':
    import glob
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    mask_rcnn_weight = "models/insulator/1/inference/model_final.pth"
    img_file = "models/000003.jpg"

    predictor = load_predictor(score_thresh=0.5)
    # for img_file in glob.glob(os.path.join("/home/yh/meter_recognition/test/test/meter","*.jpg"))[-1]:
    img = cv2.imread(img_file)
    print(type(img[0,0,0]))
    boxes, scores, classes = inference(predictor, img)

    out_file = img_file[:-4] + "_centernet.jpg"

    for b in boxes:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), thickness=2)
        cv2.imwrite(out_file, img)

    print(boxes)

        

