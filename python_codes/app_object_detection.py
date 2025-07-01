import os
import time
import cv2
import json
from lib_image_ops import base642img, img2base64, img_chinese
from lib_rcnn_ops import check_iou
from lib_inference_yolov8 import inference_yolov8
from lib_img_registration import roi_registration
import config_object_name
from config_object_name import convert_label, defect_LIST, DEFAULT_STATE
import numpy as np
from lib_help_base import GetInputData, is_include, color_list, creat_img_result, draw_region_result,reg_crop
## 表计， 二次设备，17类缺陷, 安全帽， 烟火
from config_model_list import model_type_dict,model_label_dict,model_dict
from lib_model_import import model_load
from config_model_list import model_threshold_dict
import math



def inspection_object_detection(input_data):
    """
    yolov8的目标检测推理。
    """
    ## 提取输入请求信息
    DATA = GetInputData(input_data)
    checkpoint = DATA.checkpoint; an_type = DATA.type
    img_tag = DATA.img_tag; img_ref = DATA.img_ref
    roi = DATA.roi; label_list = DATA.label_list
    reg_box=DATA.regbox
    sense = DATA.sense

    ## 初始化out_data
    out_data = {"code": 0, "data":{}, "img_result": input_data["image"], "msg": "Request; "} 

    ## 画上点位名称
    img_tag_ = img_tag.copy()
    img_tag_ = img_chinese(img_tag_, an_type + "_" + checkpoint , (10, 100), color=(255, 0, 0), size=30)

    if an_type not in model_type_dict.keys():
        out_data["msg"] = out_data["msg"] + "Type isn't object; "
        out_data["code"] = 1
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 130), color=(255, 0, 0), size=30)
        out_data["img_result"] = creat_img_result(input_data, img_tag_)  # 返回结果图
        # "data": {"no_roi": [{"label": "0", "bbox": []}]}
        # "data": {"no_roi": [{"label": "0", "label_en": "0", "bbox": [], "score": 0}]}
        out_data["data"]["no_ori"] = [{"label": "0", "label_en": "0", "bbox": [], "score": 0}]
        return out_data

    yolov8_model=model_load(an_type)
    conf = model_threshold_dict[an_type]
    model_type=model_type_dict[an_type]
    if an_type == "rec_defect":
        labels = defect_LIST
        if len(label_list) > 0:
            labels = [convert_label(l, "rec_defect") for l in label_list]
            if "jsxs" in label_list:
                labels = labels + ["jsxs_ddjt", "jsxs_ddyx", "jsxs_jdyxx", "jsxs_ecjxh"]
            if "bjdsyc" in label_list:
                labels = labels + ["bjdsyc_zz", "bjdsyc_sx", "bjdsyc_ywj", "bjdsyc_ywc"]
    elif an_type in model_label_dict.keys():
        labels=model_label_dict[an_type]
    else:
        labels_dict = yolov8_model.names
        labels = [labels_dict[id] for id in labels_dict]



    #img_ref截取regbox区域用于特征匹配
    if reg_box and len(reg_box) != 0:
        img_ref=reg_crop(img_ref,*reg_box)


    ## 求出目标图像的感兴趣区域
    roi_tag, _ = roi_registration(img_ref, img_tag, roi)
    for name, c in roi_tag.items():
        cv2.rectangle(img_tag_, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), (255,0,255), thickness=1)
        # cv2.putText(img_tag_, name, (int(c[0]), int(c[1])+10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)
        s = int((c[2] - c[0]) / 10) # 根据框子大小决定字号和线条粗细。
        img_tag_ = img_chinese(img_tag_, name, (c[0], c[1]), color=(255,0,255), size=s)

    ## 模型推理
    if an_type == "rec_defect":
        ## 根据灵敏度sense调整conf_thres阈值,sense越大，conf_thres越小
        if sense is not None:
            if sense > 5:
                conf = conf - (((sense - 5) / 5) * conf)
            else:
                conf = ((5 - sense) / 5) * (1 - conf) + conf

    cfgs = inference_yolov8(yolov8_model, img_tag, focus_labels=labels, conf_thres=conf) # inference
    cfgs = check_iou(cfgs, iou_limit=0.5) # 增加iou机制

    ## labels 列表 和 color 列表
    color_dict = {}; name_dict = {}
    for i, label in enumerate(labels):
        color_dict[label] = color_list(len(labels))[i]
        if label in config_object_name.OBJECT_MAP[model_type]:
            name_dict[label] = config_object_name.OBJECT_MAP[model_type][label]
        else:
            name_dict[label] = label

    ## 判断bbox是否在roi中
    for name, roi in roi_tag.items():
        out_data["data"][name] = []
        for cfg in cfgs:
            if is_include(cfg["coor"], roi, srate=0.5):
                c = cfg["coor"]; label = cfg["label"]
                cfg_out = {"label": name_dict[label], "bbox": c, "score": float(cfg["score"]), "label_en": cfg["label"]}
                out_data["data"][name].append(cfg_out)

                # 画出识别框
                cv2.rectangle(img_tag_, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), color_dict[label], thickness=2)
                s = int((c[2] - c[0]) / 6) # 根据框子大小决定字号和线条粗细。
                img_tag_ = img_chinese(img_tag_, name_dict[label], (c[0], c[1]), color=color_dict[label], size=s)

                if an_type != "rec_defect" and an_type != "biaoshipai":
                    break
                
        if len(out_data["data"][name]) == 0:
            out_data["data"][name] = [{}]

    if an_type == "key":
        for name, _cfgs in out_data["data"].items():
            num=len(_cfgs)
            cfg_out = {"label": 'key', "number":num, "score": 0.99}
            out_data["data"][name].append(cfg_out)
            if len(out_data["data"][name]) == 0:
                out_data["data"][name] = [{}]

    ## 指示灯若没有识别到目标物，是否也默认给定识别结果
    if DEFAULT_STATE["abnormal_code"]==0 and an_type in DEFAULT_STATE and DEFAULT_STATE[an_type] is not None:
        for name, _cfgs in out_data["data"].items():
            if len(_cfgs[0]) == 0:
                c = roi_tag[name]
                c = [int(c[0]), int(c[1]), int(c[2]), int(c[3])]
                _cfg = [{"label": DEFAULT_STATE[an_type], "bbox": c, "score": 1.0}]
                cv2.rectangle(img_tag_, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), (0,0,255), thickness=2)
                s = math.ceil((c[2] - c[0]) / 6) + 1 # 根据框子大小决定字号和线条粗细。
                img_tag_ = img_chinese(img_tag_, DEFAULT_STATE[an_type], (c[0], c[1]), color=(0,0,255), size=s)
                out_data["data"][name] = _cfg
                out_data["code"] = DEFAULT_STATE["abnormal_code"]
    
    ## 给code赋值，判断是否异常
    if an_type == "rec_defect" or an_type == "fire_smoke":
        for name, _cfg in out_data["data"].items():
            if len(_cfg[0]) > 0:
                out_data["code"] = 1
    else:
        lens = [len(_cfg[0]) for name, _cfg in out_data["data"].items()]
        if any(lens): ## 全为0，返回false
            out_data["code"] = 0
        else:
            out_data["code"] = 1
    
    ## 主从逻辑中，每个roi框都画一张图
    out_data = draw_region_result(out_data, input_data, roi_tag)
    
    ## 老版本的接口输出，"data"由字典改为list
    no_roi = [name.startswith("old_roi") for name in out_data["data"]]
    if all(no_roi): ## 全为1， 返回True
        _cfgs = []
        for name, _cfg in out_data["data"].items():
            if len(_cfg) > 0:
                _cfgs.append(_cfg[0])
        out_data["data"] = _cfgs
        if out_data["data"] == [{}]:
            out_data["data"] = []
    
    # 如果data为空，赋一个初值
    for key in out_data["data"]:
        if out_data["data"][key] == [{}]:
            out_data["data"][key] = [{"label": "0", "label_en": "0", "bbox": [], "score": 0}]
    
    img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 130), color=(255, 0, 0), size=30)

    out_data["img_result"] = creat_img_result(input_data, img_tag_) # 返回结果图
    
    return out_data

if __name__ == '__main__':
    from lib_help_base import get_save_head, save_input_data, save_output_data
    # json_file = "/data/PatrolAi/result_patrol/0330051344_鸟巢点位测试_input_data.json"
    # f = open(json_file,"r",encoding='utf-8')
    # input_data = json.load(f)
    # f.close()
    input_data = {
    "checkpoint": "切换把手方向识别",
    "image": "/data/PatrolAi/test_images/切换把手方向识别_tag.jpg",
    "config": {
        "bboxes": {"roi":{
            "切换把手0000": [
                0.33,
                0.31,
                0.65,
                0.86
            ]}
        },
        "is_region": 1,
        "img_ref": "/data/PatrolAi/test_images/切换把手方向识别_ref.jpg"
    },
    "type": "rotary_switch"
}
    
    out_data = inspection_object_detection(input_data)
    save_dir, name_head = get_save_head(input_data)
    save_input_data(input_data, save_dir, name_head, draw_img=True)
    save_output_data(out_data, save_dir, name_head)
    print("-----------------------------------------------")
    for s in out_data:
        if s != "img_result":
            print(s,":",out_data[s])
    print("----------------------------------------------")
    



