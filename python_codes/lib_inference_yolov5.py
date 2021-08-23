from yolov5.models.experimental import attempt_load
import cv2
import torch
import sys

from yolov5.utils.datasets import letterbox
from yolov5.utils.general import non_max_suppression, scale_coords
import numpy as np
sys.path.append('./yolov5')





def load_yolov5_model(model_file):
    """
    # load yolov5 FP32 model
    """
    yolov5_weights = attempt_load(model_file)
    return yolov5_weights


def inference_yolov5(model_yolov5, img, resize=640):
    """
    使用yolov5对图片做推理，返回bbox信息。
    args:
        model_yolov5: 加载后的yolov5模型，使用load_yolov5_model(model_file)加载
        img_file: 需要预测的图片
    return:
        bbox_cfg: 预测的bbox信息，json文件格式为格式为[{"label": "", "coor": [x0, y0, x1, y1]}, {}, ..]
    """
    
    # img = cv2.imread(img_file)
    img_raw = img.copy()  #由于需要resize，先拷贝一份备用。

    ## 将numpy转成yolov5格式input data.
    img = letterbox(img, new_shape=resize)[0] # resize图片
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3 x 640 x 640
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img) # numpy转tensor
    img = img.float() 
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = img.unsqueeze(0)

    ## 使用yolov5预测
    pred = model_yolov5(img)[0] # Inference

    ## 使用NMS挑选预测结果
    pred_max = non_max_suppression(pred, 0.4, 0.5)[0] # Apply NMS
    pred_max = scale_coords(img.shape[2:], pred_max, img_raw.shape).round() #bbox映射为resize之前的大小

    ## 生成bbox_cfg 的json格式，有助于人看[{"label": "", "coor": [x0, y0, x1, y1]}, {}, ..]
    labels = model_yolov5.module.names if hasattr(model_yolov5, 'module') else model_yolov5.names
    bbox_cfg = []
    for res in pred_max.numpy():
        bbox = {"label": labels[int(res[-1])], "coor": (res[:4]).tolist(), "score": res[4]}
        bbox_cfg.append(bbox)

    # lib_image_ops.draw_bboxs(img_file, bbox_cfg, is_write=True)

    return bbox_cfg

if __name__ == '__main__':
    import numpy as np
    import os
    import glob
    # model_file = 'yolov5/runs/best.pt'
    count = 0
    model_file = '/home/yh/app_meter_inference/yolov5/saved_model/best_meter.pt'
    model_yolov5 = load_yolov5_model(model_file)
    for img_file in glob.glob(os.path.join("/home/yh/meter_recognition/test/test","*.jpg")):
        # img_file = 'images/WIN_20210819_15_47_09_Pro.jpg'
        img = cv2.imread(img_file)
        bbox_cfg = inference_yolov5(model_yolov5, img, resize=640)
        for bbox in bbox_cfg:
            c = np.array(bbox["coor"],dtype=int)
            img_meter = img[c[1]:c[3], c[0]:c[2]]
            cv2.imwrite(os.path.join("/home/yh/meter_recognition/test/test/meter","meter_"+str(count).zfill(2)+".jpg"), img_meter)
            # print(bbox_cfg)
            count += 1
