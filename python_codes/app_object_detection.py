import os
import time
import cv2
import json
from lib_image_ops import base642img, img2base64, img_chinese
from lib_inference_yolov5 import load_yolov5_model, inference_yolov5, check_iou
from lib_help_base import color_list
from lib_img_registration import registration, convert_coor
import config_object_name
from config_object_name import convert_label
import numpy as np
from lib_help_base import GetInputData
from lib_help_base import is_include
## 表计， 二次设备，17类缺陷, 安全帽， 烟火

yolov5_ErCiSheBei = load_yolov5_model("/data/PatrolAi/yolov5/ErCiSheBei.pt") ## 二次设备状态
yolov5_rec_defect_x6 = load_yolov5_model("/data/PatrolAi/yolov5/18cls_rec_defect_x6.pt") # 送检18类缺陷,x6模型
yolov5_daozha = load_yolov5_model("/data/PatrolAi/yolov5/daozha_v5detect.pt")  # 加载刀闸模型
yolov5_led_color = load_yolov5_model("/data/PatrolAi/yolov5/led.pt") # led灯颜色状态模型
# yolov5_dztx = load_yolov5_model("/data/PatrolAi/yolov5/daozha_texie.pt")  # 刀闸分析模型
# yolov5_coco = load_yolov5_model("/data/PatrolAi/yolov5/coco.pt") # coco模型
# yolov5_action = load_yolov5_model("/data/PatrolAi/yolov5/action.pt") # 动作识别，倒地
# yolov5_fire_smoke = load_yolov5_model("/data/PatrolAi/yolov5/fire_smoke.pt") # 烟火模型
# yolov5_helmet = load_yolov5_model("/data/PatrolAi/yolov5/helmet.pt") # 安全帽模型
# yolov5_meter = load_yolov5_model("/data/PatrolAi/yolov5/meter.pt") # 表盘

def roi_registration(img_ref, img_tag, roi_ref):
    """
    roi框纠偏，将img_ref上的roi框纠偏匹配到img_tag上
    return:
        roi_tag: 纠偏后的roi框
    """
    H, W = img_tag.shape[:2]
    if len(roi_ref) == 0:
        return [[0,0,W,H]]
    
    M = registration(img_ref, img_tag) # 求偏移矩阵

    if M is None:
        return roi_ref
    
    roi_tag = []
    for roi in roi_ref:
        coors = [(roi[0],roi[1]), (roi[2],roi[1]), (roi[2],roi[3]), (roi[0],roi[3])]
        coors_ = [list(convert_coor(coor, M)) for coor in coors]
        c_ = np.array(coors_, dtype=int)
        r = [min(c_[:,0]), min(c_[:, 1]), max(c_[:,0]), max(c_[:,1])]
        roi_tag.append([max(0, r[0]), max(0, r[1]), min(W, r[2]), min(H, r[3])])

    return roi_tag

def inspection_object_detection(input_data):
    """
    yolov5的目标检测推理。
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
    img_tag_ = img_chinese(img_tag_, checkpoint + an_type , (10, 10), color=(255, 0, 0), size=60)

    if an_type == "fire_smoke":
        yolov5_model = yolov5_fire_smoke
        labels_dict = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
        labels = [labels_dict[id] for id in labels_dict]
        model_type = "fire_smoke"
    elif an_type == "helmet":
        yolov5_model = yolov5_helmet
        labels_dict = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
        labels = [labels_dict[id] for id in labels_dict]
        model_type = "helmet"
    elif an_type == "led_color":
        yolov5_model = yolov5_led_color
        labels_dict = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
        labels = [labels_dict[id] for id in labels_dict]
        model_type = "led"
    elif an_type == "rec_defect":
        yolov5_model = yolov5_rec_defect_x6
        labels_dict = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
        labels = [labels_dict[id] for id in labels_dict]
        if len(label_list) > 0:
            labels = [convert_label(l, "rec_defect") for l in label_list]
        model_type = "rec_defect"
    elif an_type == "disconnector_notemp":
        yolov5_model = yolov5_daozha
        labels_dict = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
        labels = [labels_dict[id] for id in labels_dict]
        model_type = "disconnector_texie"
    elif an_type == "disconnector_texie":
        yolov5_model = yolov5_dztx
        labels_dict = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
        labels = [labels_dict[id] for id in labels_dict]
        model_type = "disconnector_texie"
    elif an_type == "person":
        yolov5_model = yolov5_coco
        labels = ["person"]
        model_type = "coco"
    elif an_type == "pressplate": 
        yolov5_model = yolov5_ErCiSheBei
        labels = ["kgg_ybh", "kgg_ybf"]
        model_type = "ErCiSheBei"
    elif an_type == "air_switch":
        yolov5_model = yolov5_ErCiSheBei
        labels = ["kqkg_hz", "kqkg_fz"]
        model_type = "ErCiSheBei"
    elif an_type == "led":
        yolov5_model = yolov5_ErCiSheBei
        labels = ["zsd_l", "zsd_m"]
        model_type = "ErCiSheBei"
    elif an_type == "fanpaiqi":
        yolov5_model = yolov5_ErCiSheBei
        model_type = "ErCiSheBei"
        labels = ["fpq_h", "fpq_f", "fpq_jd"]
    elif an_type == "rotary_switch":
        yolov5_model = yolov5_ErCiSheBei
        labels = ["xnkg_s", "xnkg_zs", "xnkg_ys", "xnkg_z"]
        model_type = "ErCiSheBei"
    elif an_type == "door":
        yolov5_model = yolov5_ErCiSheBei
        labels = ["xmbhyc", "xmbhzc"]
        model_type = "ErCiSheBei"
    elif an_type == "key":
        yolov5_model = yolov5_ErCiSheBei
        labels = ["ys"]
        model_type = "ErCiSheBei"
    else:
        out_data["msg"] = out_data["msg"] + "Type isn't object; "
        out_data["code"] = 1
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
        out_data["img_result"] = img2base64(img_tag_)
        return out_data

    ## 求出目标图像的感兴趣区域
    roi_tag = roi_registration(img_ref, img_tag, roi)
    for c in roi_tag:
        cv2.rectangle(img_tag_, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), (255,0,255), thickness=1)
        cv2.putText(img_tag_, "roi", (int(c[0]), int(c[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)

    ## 模型推理
    if an_type == "rec_defect":
        cfgs = inference_yolov5(yolov5_model, img_tag, resize=1280, pre_labels=labels, conf_thres=0.7) # inference
    else:
        cfgs = inference_yolov5(yolov5_model, img_tag, resize=640, pre_labels=labels, conf_thres=0.3) # inference
    cfgs = check_iou(cfgs, iou_limit=0.5) # 增加iou机制

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
        for cfg in cfgs:
            if is_include(cfg["coor"], roi, srate=0.5):
                cfg_out = {"label": name_dict[cfg["label"]], "bbox": cfg["coor"], "score": float(cfg["score"])}
                break
            else:
                cfg_out = {}
        out_data["data"].append(cfg_out)
    
    ## 判断是否异常
    if an_type == "rec_defect" or an_type == "fire_smoke":
        for _cfg in out_data["data"]:
            if len(_cfg) > 0:
                out_data["code"] = 1
    else:
        for _cfg in out_data["data"]:
            if len(_cfg) == 0:
                out_data["code"] = 1
    
    img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
    out_data["img_result"] = img2base64(img_tag_)
    
    return out_data

if __name__ == '__main__':
    from lib_help_base import get_save_head, save_input_data, save_output_data
    json_file = "/data/PatrolAi/result_patrol/0627082120__input_data.json"
    f = open(json_file,"r",encoding='utf-8')
    input_data = json.load(f)
    f.close()
    
    out_data = inspection_object_detection(input_data)
    save_dir, name_head = get_save_head(input_data)
    save_input_data(input_data, save_dir, name_head, draw_img=True)
    save_output_data(out_data, save_dir, name_head)
    print("-----------------------------------------------")
    for s in out_data:
        if s != "img_result":
            print(s,":",out_data[s])
    print("----------------------------------------------")
    



