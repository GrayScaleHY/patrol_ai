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
from utils.torch_utils import select_device
from models.experimental import attempt_load  # scoped to avoid circular import
from lib_rcnn_ops import iou, filter_cfgs

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

    
def load_yolov5_model(model_file):
    """
    load yolov5 FP32 model
    """
    yolov5_weights = attempt_load(model_file, device)  # 加载模型
    return yolov5_weights

def inference_yolov5(model,
                     img,
                     resize=640,
                     conf_thres=0.25,
                     same_iou_thres=0.7,
                     diff_iou_thres=1,
                     pre_labels=None
                     ):
    """
    使用yolov5对图片做推理，返回bbox信息。
    args:
        model: 加载后的yolov5模型，使用load_yolov5_model(model_file)加载
        img: 需要预测的图片
        resize: 送检模型里的图片大小
        conf_thres: 置信度阈值
        same_iou_thres: 同类标签之间的iou阈值。
        diff_iou_thres: 所有目标物之间的iou阈值。
        pre_labels: 关注的标签。若pre_labels=None,则不过滤；若pre_labels为list，则过滤pre_labels以外的标签。
    return:
        cfgs: 预测的bbox信息，json文件格式为格式为[{"label": "", "coor": [x0, y0, x1, y1], "score": float}, {}, ..]
    """
    img_raw_shape = img.shape  # 记录原图大小。

    ## 将numpy转成yolov5格式input data. 
    stride = max(int(model.stride.max()), 32)
    img = letterbox(img, new_shape=resize, auto=True, stride=stride)[0] # resize图片
    img_resize_shape = img.shape # 记录resize后图片大小
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3 x 640 x 640
    img = torch.from_numpy(img.copy()).to(device) # numpy转tenso
    img = img.float()
    img /= 255  # 0 - 255 to 0.0 - 1.0
    img = img.unsqueeze(0) # 添加一维

    ## 使用yolov5预测
    pred = model(img, augment=False, visualize=False)[0] # Inference

    ## 使用NMS挑选预测结果
    pred_max = non_max_suppression(pred, conf_thres, same_iou_thres)[0] # Apply NMS
    pred_max = scale_boxes(img_resize_shape, pred_max, img_raw_shape) #bbox映射为resize之前的大小

    ## 生成cfgs 的json格式，有助于人看[{"label": "", "coor": [x0, y0, x1, y1]}, {}, ..]
    labels = model.module.names if hasattr(model, 'module') else model.names
    cfgs = []
    for res in pred_max.cpu().numpy():
        label = labels[int(res[-1])]
        bbox = {"label": label, "coor": (res[:4]).astype(int).tolist(), "score": res[4]}
        if pre_labels is None or label in pre_labels:
            cfgs.append(bbox)
    
    if len(cfgs) == 0:
        return []
        
    # 根据conf_thres、iou_thres、focus_labels过滤结果
    if diff_iou_thres < 1:
        cfgs = filter_cfgs(cfgs, conf_thres, same_iou_thres,
                        diff_iou_thres, focus_labels=pre_labels)

    return cfgs

if __name__ == '__main__':

    img_file = "11-06-17-10-35_img_tag.jpg"
    weight = "best.pt"
    img = cv2.imread(img_file)
    model = load_yolov5_model(weight)
    cfgs = inference_yolov5(model, img)

    for i in range(len(cfgs)):
        tmp = cfgs[i]
        c = tmp["coor"]
        label = tmp["label"]
        score = tmp["score"]
        cv2.rectangle(img, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0, 255, 0), 2)
        cv2.putText(img, label + ": " + str(score), (int(c[0]), int(c[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 255), thickness=2)
    cv2.imwrite('33.jpg', img)


                


