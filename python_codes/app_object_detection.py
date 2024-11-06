import os
import time
import cv2
import json
from lib_image_ops import base642img, img2base64, img_chinese
from lib_rcnn_ops import check_iou
from lib_inference_yolov8 import load_yolov8_model, inference_yolov8
from lib_img_registration import roi_registration
import config_object_name
from config_object_name import convert_label, defect_LIST, DEFAULT_STATE
import numpy as np
from lib_help_base import GetInputData, is_include, color_list, creat_img_result
## 表计， 二次设备，17类缺陷, 安全帽， 烟火

yolov8_ErCiSheBei = load_yolov8_model("/data/PatrolAi/yolov8/ErCiSheBei.pt") ## 二次设备状态
yolov8_rec_defect = load_yolov8_model("/data/PatrolAi/yolov8/rec_defect.pt") # 送检18类缺陷,x6模型
yolov8_daozha = load_yolov8_model("/data/PatrolAi/yolov8/daozha_v5detect.pt")  # 加载刀闸模型
yolov8_led_color = load_yolov8_model("/data/PatrolAi/yolov8/led.pt") # led灯颜色状态模型
# yolov8_count = load_yolov8_model("/data/PatrolAi/yolov8/count.pt")#大电流端子借用钥匙计数功能
# yolov8_dztx = load_yolov8_model("/data/PatrolAi/yolov8/daozha_texie.pt")  # 刀闸分析模型
# yolov8_coco = load_yolov8_model("/data/PatrolAi/yolov8/coco.pt") # coco模型
# yolov8_action = load_yolov8_model("/data/PatrolAi/yolov8/action.pt") # 动作识别，倒地
# yolov8_fire_smoke = load_yolov8_model("/data/PatrolAi/yolov8/fire_smoke.pt") # 烟火模型
# yolov8_helmet = load_yolov8_model("/data/PatrolAi/yolov8/helmet.pt") # 安全帽模型
# yolov8_meter = load_yolov8_model("/data/PatrolAi/yolov8/meter.pt") # 表盘
# yolov8_biaoshipai = load_yolov8_model("/data/PatrolAi/yolov8/biaoshipai.pt") # 表盘

def inspection_object_detection(input_data):
    """
    yolov8的目标检测推理。
    """
    ## 提取输入请求信息
    DATA = GetInputData(input_data)
    checkpoint = DATA.checkpoint; an_type = DATA.type
    img_tag = DATA.img_tag; img_ref = DATA.img_ref
    roi = DATA.roi; label_list = DATA.label_list
    sense = DATA.sense

    ## 初始化out_data
    out_data = {"code": 0, "data":{}, "img_result": input_data["image"], "msg": "Request; "} 

    ## 画上点位名称
    img_tag_ = img_tag.copy()
    img_tag_ = img_chinese(img_tag_, an_type + "_" + checkpoint , (10, 100), color=(255, 0, 0), size=30)

    if an_type == "fire_smoke":
        yolov8_model = yolov8_fire_smoke
        labels_dict = yolov8_model.names
        labels = [labels_dict[id] for id in labels_dict]
        model_type = "fire_smoke"
    elif an_type == "helmet":
        yolov8_model = yolov8_helmet
        labels_dict = yolov8_model.names
        labels = [labels_dict[id] for id in labels_dict]
        model_type = "helmet"
    elif an_type == "biaoshipai":
        yolov8_model = yolov8_biaoshipai
        labels_dict = yolov8_model.names
        labels = [labels_dict[id] for id in labels_dict]
        model_type = "biaoshipai"
    elif an_type == "led_color":
        yolov8_model = yolov8_led_color
        labels_dict = yolov8_model.names
        labels = [labels_dict[id] for id in labels_dict]
        model_type = "led"
    elif an_type == "rec_defect":
        yolov8_model = yolov8_rec_defect
        labels = defect_LIST
        if len(label_list) > 0:
            labels = [convert_label(l, "rec_defect") for l in label_list]
            if "jsxs" in label_list:
                labels = labels + ["jsxs_ddjt", "jsxs_ddyx", "jsxs_jdyxx", "jsxs_ecjxh"]
            if "bjdsyc" in label_list:
                labels = labels + ["bjdsyc_zz", "bjdsyc_sx", "bjdsyc_ywj", "bjdsyc_ywc"]
        model_type = "rec_defect"
    elif an_type == "disconnector_notemp":
        yolov8_model = yolov8_daozha
        labels_dict = yolov8_model.names
        labels = ["he","fen","budaowei"]
        model_type = "disconnector_texie"
    elif an_type == "disconnector_texie":
        yolov8_model = yolov8_dztx
        labels_dict = yolov8_model.names
        labels = ["he","fen","budaowei"]
        model_type = "disconnector_texie"
    elif an_type == "person":
        yolov8_model = yolov8_coco
        labels = ["person"]
        model_type = "coco"
    elif an_type == "pressplate": 
        yolov8_model = yolov8_ErCiSheBei
        labels = ["kgg_ybh", "kgg_ybf"]
        model_type = "ErCiSheBei"
    elif an_type == "air_switch":
        yolov8_model = yolov8_ErCiSheBei
        labels = ["kqkg_hz", "kqkg_fz"]
        model_type = "ErCiSheBei"
    elif an_type == "led":
        yolov8_model = yolov8_ErCiSheBei
        labels = ["zsd_l", "zsd_m"]
        model_type = "ErCiSheBei"
    elif an_type == "fanpaiqi":
        yolov8_model = yolov8_ErCiSheBei
        model_type = "ErCiSheBei"
        labels = ["fpq_h", "fpq_f", "fpq_jd"]
    elif an_type == "rotary_switch":
        yolov8_model = yolov8_ErCiSheBei
        labels = ["xnkg_s", "xnkg_zs", "xnkg_ys", "xnkg_z"]
        model_type = "ErCiSheBei"
    elif an_type == "door":
        yolov8_model = yolov8_ErCiSheBei
        labels = ["xmbhyc", "xmbhzc"]
        model_type = "ErCiSheBei"
    elif an_type == "key":
        yolov8_model = yolov8_count
        labels = ["ys"]
        model_type = "ErCiSheBei"
    else:
        out_data["msg"] = out_data["msg"] + "Type isn't object; "
        out_data["code"] = 1
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 130), color=(255, 0, 0), size=30)
        out_data["img_result"] = creat_img_result(input_data, img_tag_) # 返回结果图
        return out_data

    ## 求出目标图像的感兴趣区域
    roi_tag, _ = roi_registration(img_ref, img_tag, roi)
    for name, c in roi_tag.items():
        cv2.rectangle(img_tag_, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), (255,0,255), thickness=1)
        cv2.putText(img_tag_, name, (int(c[0]), int(c[1])+10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)

    ## 模型推理
    conf_thres = 0.3
    if an_type == "rec_defect":
        conf_thres = 0.3
        ## 根据灵敏度sense调整conf_thres阈值,sense越大，conf_thres越小
        if sense is not None:
            if sense > 5:
                conf_thres = conf_thres - (((sense - 5) / 5) * conf_thres)
            else:
                conf_thres = ((5 - sense) / 5) * (1 - conf_thres) + conf_thres

    cfgs = inference_yolov8(yolov8_model, img_tag, focus_labels=labels, conf_thres=conf_thres) # inference
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
    if an_type in DEFAULT_STATE and DEFAULT_STATE[an_type] is not None:
        for name, _cfgs in out_data["data"].items():
            if len(_cfgs[0]) == 0:
                c = roi_tag[name]
                c = [int(c[0]), int(c[1]), int(c[2]), int(c[3])]
                _cfg = [{"label": DEFAULT_STATE[an_type], "bbox": c, "score": 1.0}]
                cv2.rectangle(img_tag_, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), (0,0,255), thickness=2)
                s = int((c[2] - c[0]) / 6) # 根据框子大小决定字号和线条粗细。
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
    "checkpoint": "巡视点位01",  

    "image":"/data/PatrolAi/patrol_ai/python_codes/images/014237_001_10101_16392_3698003230.jpg", 
   
    "type": "biaoshipai"

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
    



