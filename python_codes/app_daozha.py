import cv2
import json
from lib_image_ops import base642img, img2base64, img_chinese
from lib_inference_yolov5 import load_yolov5_model, inference_yolov5, check_iou
from lib_help_base import color_list
from lib_img_registration import roi_registration
import config_object_name
import numpy as np
from lib_help_base import GetInputData
from lib_help_base import is_include

yolov5_daozha = load_yolov5_model("/data/PatrolAi/yolov5/daozha_v5detect.pt")
yolov5_dztx = load_yolov5_model("/data/PatrolAi/yolov5/daozha_texie.pt")  # 刀闸分析模型

def patrol_daozha(input_data):
    """
    刀闸状态检测
    """
    ## 提取输入请求信息
    DATA = GetInputData(input_data)
    checkpoint = DATA.checkpoint; an_type = DATA.type
    img_tag = DATA.img_tag; img_ref = DATA.img_ref
    roi = DATA.roi; label_list = DATA.label_list

    ## 初始化out_data
    out_data = {"code": 0, "data":[], "img_result": input_data["image"], "msg": "Request; "} 

    ## 画上点位名称
    img_tag_ = img_tag.copy()
    img_tag_ = img_chinese(img_tag_, an_type + "_" + checkpoint , (10, 100), color=(255, 0, 0), size=30)

    if an_type == "disconnector_notemp":
        yolov5_model = yolov5_daozha
        labels = ["he","fen","budaowei"]
        model_type = "disconnector_texie"
    elif an_type == "disconnector_texie":
        yolov5_model = yolov5_dztx
        labels = ["he","fen","budaowei"]
        model_type = "disconnector_texie"
    else:
        out_data["msg"] = out_data["msg"] + "Type isn't object; "
        out_data["code"] = 1
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 130), color=(255, 0, 0), size=30)
        out_data["img_result"] = img2base64(img_tag_)
        return out_data
    
    ## 求出目标图像的感兴趣区域
    roi_tag = roi_registration(img_ref, img_tag, roi)
    for c in roi_tag:
        cv2.rectangle(img_tag_, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), (255,0,255), thickness=1)
        cv2.putText(img_tag_, "roi", (int(c[0]), int(c[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)

    ## 模型推理
    cfgs = inference_yolov5(yolov5_model, img_tag, pre_labels=labels, same_iou_thres=0.5, diff_iou_thres=0.5)

    ## labels 列表 和 color 列表
    color_dict = {}; name_dict = {}
    for i, label in enumerate(labels):
        color_dict[label] = color_list(len(labels))[i]
        if label in config_object_name.OBJECT_MAP[model_type]:
            name_dict[label] = config_object_name.OBJECT_MAP[model_type][label]
        else:
            name_dict[label] = label

    ## 画出boxes
    for cfg in cfgs:
        c = cfg["coor"]; label = cfg["label"]
        cv2.rectangle(img_tag_, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), color_dict[label], thickness=2)
        s = int((c[2] - c[0]) / 6) # 根据框子大小决定字号和线条粗细。
        img_tag_ = img_chinese(img_tag_, name_dict[label], (c[0], c[1]), color=color_dict[label], size=s)

    
    ## 判断bbox是否在roi中
    for roi in roi_tag:
        cfg_out = {}
        for cfg in cfgs:
            if is_include(cfg["coor"], roi, srate=0.3):
                cfg_out = {"label": name_dict[cfg["label"]], "bbox": cfg["coor"], "score": float(cfg["score"])}
                break
            else:
                cfg_out = {}
        out_data["data"].append(cfg_out)

    ## 判断是否异常
    for i, _cfg in enumerate(out_data["data"]):
        if len(_cfg) == 0 or _cfg["label"] == "分合异常":
            out_data["code"] = 1  ## 报异常的情况
            out_data["data"][i] = {"label": "分合异常", "bbox": roi_tag[i], "score": 1.0}

    img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 130), color=(255, 0, 0), size=30)
    out_data["img_result"] = img2base64(img_tag_)

    return out_data



