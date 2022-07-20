import cv2
import torch
import sys
import time
import os
import glob

sys.path.append('../yolov5') ## ultralytics/yolov5 存放的路径
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from models.common import DetectMultiBackend
import numpy as np
from utils.torch_utils import select_device
from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

device = select_device("0")  ## 选择gpu: 'cpu' or '0' or '0,1,2,3'

def load_yolov5_model(model_file):
    """
    # load yolov5 FP32 model
    """
    # yolov5_weights = DetectMultiBackend(model_file, device=device) #, dnn=False, data='data/coco128.yaml', fp16=False
    yolov5_weights = attempt_load(model_file , map_location=device) # 加载模型
    return yolov5_weights

def inference_yolov5(model_yolov5, img, resize=640, conf_thres=0.2, iou_thres=0.2, pre_labels=None):
    """
    使用yolov5对图片做推理，返回bbox信息。
    args:
        model_yolov5: 加载后的yolov5模型，使用load_yolov5_model(model_file)加载
        img_file: 需要预测的图片
    return:
        bbox_cfg: 预测的bbox信息，json文件格式为格式为[{"label": "", "coor": [x0, y0, x1, y1], "score": float}, {}, ..]
    """
    
    # img = cv2.imread(img_file)
    img_raw = img.copy()  #由于需要resize，先拷贝一份备用。

    ## 将numpy转成yolov5格式input data. 
    img_resize = letterbox(img, new_shape=resize)[0] # resize图片
    img_zeros = np.zeros([resize, resize, 3], dtype=np.uint8) 
    img_zeros[:img_resize.shape[0], :img_resize.shape[1], :img_resize.shape[2]] = img_resize
    img = img_zeros # 将图片resize成正方形
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3 x 640 x 640
    img = torch.from_numpy(img.copy()).to(device) # numpy转tenso
    img = img.float()
    img /= 255  # 0 - 255 to 0.0 - 1.0
    img = img.unsqueeze(0) # 添加一维
    # if len(img.shape) == 3:
    #     img = img[None]  # expand for batch dim

    ## 使用yolov5预测
    pred = model_yolov5(img, augment=False, visualize=False)[0] # Inference

    ## 使用NMS挑选预测结果
    pred_max = non_max_suppression(pred, conf_thres, iou_thres)[0] # Apply NMS
    pred_max = scale_coords(img_resize.shape, pred_max, img_raw.shape) #bbox映射为resize之前的大小

    ## 生成bbox_cfg 的json格式，有助于人看[{"label": "", "coor": [x0, y0, x1, y1]}, {}, ..]
    labels = model_yolov5.module.names if hasattr(model_yolov5, 'module') else model_yolov5.names
    bbox_cfg = []
    for res in pred_max.cpu().numpy():
        label = labels[int(res[-1])]
        bbox = {"label": label, "coor": (res[:4]).astype(int).tolist(), "score": res[4]}
        if pre_labels is None or label in pre_labels:
            bbox_cfg.append(bbox)
    # lib_image_ops.draw_bboxs(img_file, bbox_cfg, is_write=True)

    return bbox_cfg


if __name__ == '__main__':
    import shutil
    img_file = "/home/yh/image/python_codes/test/source_0.bmp"
    weight = "/data/PatrolAi/yolov5/rec_defect_x6.pt"
    img = cv2.imread(img_file)
    model_yolov5 = load_yolov5_model(weight)
    cfgs = inference_yolov5(model_yolov5, img, resize=1280, conf_thres=0.1, iou_thres=0.2)
    for cfg in cfgs:
        c = cfg["coor"]; label = cfg["label"]; score = cfg["score"]
        cv2.rectangle(img, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), (255,0,255), thickness=2)
        cv2.putText(img, label+": "+str(score), (int(c[0]), int(c[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), thickness=2)
    cv2.imwrite(img_file[:-4] + "result.jpg", img)
    print(cfgs)


                


