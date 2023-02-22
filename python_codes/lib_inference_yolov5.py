import cv2
import torch
import sys
import time
import os
import glob

sys.path.insert(0,'../yolov5') ## ultralytics/yolov5 存放的路径
try:
    from utils.dataloaders import letterbox ## v7.0
except:
    from utils.datasets import letterbox ## v6.0
from utils.general import non_max_suppression
try:
    from utils.general import scale_coords as scale_boxes # v6.2
except:
    from utils.general import scale_boxes # v7.0

from models.common import DetectMultiBackend
import numpy as np
from utils.torch_utils import select_device
from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import
from lib_rcnn_ops import iou
from utils.segment.general import masks2segments, process_mask
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer, xyxy2xywh)

device = select_device("0")  ## 选择gpu: 'cpu' or '0' or '0,1,2,3'

def check_iou(cfgs_in, iou_limit=0.8):
    """
    将模型推理结果中重合度高的框去掉（即使label不一样）。
    args:
        cfgs_in: inference_yolov5的输出,格式为[{"label": "", "coor": [x0, y0, x1, y1], "score": float}, {}, ..]
        iou_limit: iou阈值
    return:
        cfgs_out: 去重后的cfgs
    """
    if len(cfgs_in) < 2:
        return cfgs_in

    rm_ids = []
    for i in range(len(cfgs_in)):
        for j in range(len(cfgs_in)):
            if j == i:
                continue
            if iou(cfgs_in[i]["coor"], cfgs_in[j]["coor"]) < iou_limit:
                continue
        
            if cfgs_in[i]["score"] > cfgs_in[j]["score"]:
                rm_ids.append(j)
            elif cfgs_in[i]["score"] > cfgs_in[j]["score"]:
                continue
            else:
                rm_ids.append(i)
    rm_ids = list(set(rm_ids))

    cfgs_out = [c for i, c in enumerate(cfgs_in) if i not in rm_ids]
    return cfgs_out
    
def load_yolov5_model(model_file,decode=False):
    """
    # load yolov5 FP32 model
    """
    # yolov5_weights = DetectMultiBackend(model_file, device=device) #, dnn=False, data='data/coco128.yaml', fp16=False
    '''from lib_decode_model import decode_model
    if decode:
        model_file = decode_model(model_file)[0]
        yolov5_weights = attempt_load(model_file, device) # 加载模型
        os.remove(model_file)
    else:
        yolov5_weights = attempt_load(model_file, device) # 加载模型
    '''
    try:
        yolov5_weights = attempt_load(model_file, device)  # 加载模型
    except:
        from lib_decode_model import decode_model
        model_file = decode_model(model_file)[0]
        yolov5_weights = attempt_load(model_file, device)  # 加载模型
        os.remove(model_file)
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
    img_raw_shape = img.shape  # 记录原图大小。

    ## 将numpy转成yolov5格式input data. 
    stride = max(int(model_yolov5.stride.max()), 32)
    img = letterbox(img, new_shape=resize, auto=True, stride=stride)[0] # resize图片
    img_resize_shape = img.shape # 记录resize后图片大小
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3 x 640 x 640
    img = torch.from_numpy(img.copy()).to(device) # numpy转tenso
    img = img.float()
    img /= 255  # 0 - 255 to 0.0 - 1.0
    img = img.unsqueeze(0) # 添加一维

    ## 使用yolov5预测
    pred = model_yolov5(img, augment=False, visualize=False)[0] # Inference

    ## 使用NMS挑选预测结果
    pred_max = non_max_suppression(pred, conf_thres, iou_thres)[0] # Apply NMS
    pred_max = scale_boxes(img_resize_shape, pred_max, img_raw_shape) #bbox映射为resize之前的大小

    ## 生成bbox_cfg 的json格式，有助于人看[{"label": "", "coor": [x0, y0, x1, y1]}, {}, ..]
    labels = model_yolov5.module.names if hasattr(model_yolov5, 'module') else model_yolov5.names
    bbox_cfg = []
    for res in pred_max.cpu().numpy():
        label = labels[int(res[-1])]
        bbox = {"label": label, "coor": (res[:4]).astype(int).tolist(), "score": res[4]}
        if pre_labels is None or label in pre_labels:
            bbox_cfg.append(bbox)

    return bbox_cfg


def load_yolov5seg_model(model_file='/data/PatrolAi/yolov5/daozha_seg.pt',decode=False):
    """
    # load yolov5 FP32 model
    """
    '''from lib_decode_model import decode_model
    if decode:
        model_file = decode_model(model_file)[0]
        yolov5_weights = DetectMultiBackend(model_file, device=device) #, dnn=False, data='data/coco128.yaml', fp16=False
        os.remove(model_file)
    else:
        yolov5_weights = DetectMultiBackend(model_file,device=device)'''
    # yolov5_weights = attempt_load(model_file, device) # 加载模型
    try:
        yolov5_weights = DetectMultiBackend(model_file, device)  # 加载模型
    except:
        from lib_decode_model import decode_model
        model_file = decode_model(model_file)[0]
        yolov5_weights = DetectMultiBackend(model_file, device)  # 加载模型
        os.remove(model_file)
    return yolov5_weights

def inference_yolov5seg(model_yolov5, img, resize=640, conf_thres=0.2, iou_thres=0.2, pre_labels=None):
    """
    使用yolov5对图片做推理，返回bbox信息。
    args:
        model_yolov5: 加载后的yolov5模型，使用load_yolov5_model_seg(model_file)加载
        img_file: 需要预测的图片
    return:
        bbox_cfg: 预测的bbox信息，json文件格式为格式为[{"label": "", "coor": [x0, y0, x1, y1], "score": float}, {}, ..]
    """
    model=model_yolov5
    imgsz=(resize,resize)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Run inference
    bs = 1  # batch_size
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    img_raw_shape = img.shape  # 记录原图大小。

    ## 将numpy转成yolov5格式input data.
    stride = max(int(model_yolov5.stride), 32)
    img = letterbox(img, new_shape=resize, auto=True, stride=stride)[0]  # resize图片
    img_resize_shape = img.shape  # 记录resize后图片大小
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3 x 640 x 640
    img = torch.from_numpy(img.copy()).to(model.device)  # numpy转tenso
    img = img.float()
    img /= 255  # 0 - 255 to 0.0 - 1.0
    im = img.unsqueeze(0)  # 添加一维

    print(im.shape)

    # Inference
    with dt[1]:
        pred, proto = model(im, augment=False, visualize=False)[:2]

    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, max_det=1000, nm=32)

    ## 生成bbox_cfg 的json格式，有助于人看[{"label": "", "coor": [x0, y0, x1, y1]}, {}, ..]
    labels = model_yolov5.module.names if hasattr(model_yolov5, 'module') else model_yolov5.names
    bbox_cfg = []
    # Process predictions
    for i, det in enumerate(pred):  # per image
        print('det shape',det.shape)
        if len(det):
            masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC

            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img_raw_shape).round()  # rescale boxes to im0 size
            segments = reversed(masks2segments(masks))
            segments = [scale_segments(im.shape[2:], x, img_raw_shape, normalize=True) for x in segments]

            print('inf', masks.shape)

            bbox=det[:, :4]
            score=det[:,4]
            classes=det[:,5]

            for i in range(len(det.cpu().numpy())):
                label = labels[int(classes[i])]
                print('maski',masks[i].shape)
                segment = np.array(segments[i])
                segment[:, 0] = segment[:, 0] * img_raw_shape[1]
                segment[:, 1] = segment[:, 1] * img_raw_shape[0]
                segment = segment.astype(int)
                tmp = {"label": label, "coor": bbox[i].cpu().numpy().astype(int).tolist(), "score": float(score[i]),"mask":masks[i],'segment':segment}
                if pre_labels is None or label in pre_labels:
                    bbox_cfg.append(tmp)

    return bbox_cfg

if __name__ == '__main__':
    import shutil

    img_file = "11-06-17-10-35_img_tag.jpg"
    weight = "best.pt"
    img = cv2.imread(img_file)
    model = load_yolov5seg_model()
    bbox_cfg = inference_yolov5seg(model, img)

    for i in range(len(bbox_cfg)):
        tmp = bbox_cfg[i]
        c = tmp["coor"];
        label = tmp["label"];
        score = tmp["score"]
        mask = tmp['mask']
        print(mask)
        cv2.polylines(img, [mask], isClosed=True, color=(0, 0, 255), thickness=1)
        cv2.rectangle(img, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0, 255, 0), 2)
        cv2.putText(img, label + ": " + str(score), (int(c[0]), int(c[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 255), thickness=2)
    cv2.imwrite('33.jpg', img)


                


