import os
import time
import cv2
import json
from lib_image_ops import base642img, img2base64, img_chinese
from lib_inference_yolov5 import load_yolov5_model, inference_yolov5, check_iou
from lib_help_base import color_list
from lib_sift_match import sift_match, convert_coor, sift_create, cupy_affine
import config_object_name
from config_object_name import convert_label
import numpy as np
from lib_help_base import GetInputData
from lib_help_base import is_include
## 表计， 二次设备，17类缺陷, 安全帽， 烟火
from config_load_models_var import yolov5_meter, \
                                   yolov5_ErCiSheBei, \
                                   yolov5_rec_defect_x6, \
                                   yolov5_coco, \
                                   yolov5_ShuZiBiaoJi, \
                                   yolov5_dztx
                                #    yolov5_jmjs, \
                                #    yolov5_helmet, \
                                #    yolov5_fire_smoke, \
                                #    yolov5_led_color


def rank_digital(obj_data, obj_type="counter"):
    """
    args:
        obj_data: 常规目标检测输出格式，[{"label": "0", "bbox": [xmin,ymin,xmax,ymax], "score":0.635}, ..]
        obj_type: counter or digital
    return:
        new_data: 数字类排好序的格式，{"type": "counter", "values": ['6', '5'], "bboxes": [[xmin,ymin,xmax,ymax], ..]}
    """
    l = [cfg["bbox"][0] for cfg in obj_data]
    rank = [index for index,value in sorted(list(enumerate(l)),key=lambda x:x[1])]
    vals = [obj_data[i]["label"] for i in rank]
    bboxes = [obj_data[i]["bbox"] for i in rank]
    new_data = {"type": obj_type, "values": vals, "bboxes": bboxes}
    return new_data

def inspection_object_detection(input_data):
    """
    yolov5的目标检测推理。
    """
    ## 提取输入请求信息
    DATA = GetInputData(input_data)
    checkpoint = DATA.checkpoint; an_type = DATA.type
    img_tag = DATA.img_tag; img_ref = DATA.img_ref
    roi = DATA.roi; osd = DATA.osd
    status_map = DATA.status_map; label_list = DATA.label_list

    ## 初始化out_data
    out_data = {"code": 0, "data":[], "img_result": input_data["image"], "msg": "Request; "} 
    if input_data["type"] == "rec_defect" or input_data["type"] == "fire_smoke":
        out_data["code"] = 1

    ## 画上点位名称和osd区域
    img_tag_ = img_tag.copy()
    img_tag_ = img_chinese(img_tag_, checkpoint + an_type , (10, 10), color=(255, 0, 0), size=60)
    for o_ in osd:  ## 如果配置了感兴趣区域，则画出osd区域
        cv2.rectangle(img_tag_, (int(o_[0]), int(o_[1])),(int(o_[2]), int(o_[3])), (255, 0, 255), thickness=1)
        cv2.putText(img_tag_, "osd", (int(o_[0]), int(o_[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)

    ## 选择模型
    if input_data["type"] == "meter":
        yolov5_model = yolov5_meter
        labels_dict = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
        labels = [labels_dict[id] for id in labels_dict]
        model_type = "meter"
    # elif input_data["type"] == "fire_smoke":
    #     yolov5_model = yolov5_fire_smoke
    #     labels_dict = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
    #     labels = [labels_dict[id] for id in labels_dict]
    #     model_type = "fire_smoke"
    # elif input_data["type"] == "helmet":
    #     yolov5_model = yolov5_helmet
    #     labels_dict = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
    #     labels = [labels_dict[id] for id in labels_dict]
    #     model_type = "helmet"
    # elif input_data["type"] == "led_color":
    #     yolov5_model = yolov5_led_color
    #     labels_dict = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
    #     labels = [labels_dict[id] for id in labels_dict]
    #     model_type = "led"
    elif input_data["type"] == "digital":
        yolov5_model = yolov5_ShuZiBiaoJi
        labels_dict = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
        labels = [labels_dict[id] for id in labels_dict]
        model_type = "digital"
    elif input_data["type"] == "counter":
        yolov5_model = yolov5_ShuZiBiaoJi
        labels = ["0","1","2","3","4","5","6","7","8","9"]
        model_type = "counter"
    elif input_data["type"] == "rec_defect":
        if label_list == ["xdwcr"]:
            yolov5_model = yolov5_coco
            labels = ["bird", "cat", "dog", "sheep"]
            model_type = "meter"
        # elif label_list == ["hzyw"]:
        #     yolov5_model = yolov5_fire_smoke
        #     labels_dict = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
        #     labels = [labels_dict[id] for id in labels_dict]
        #     model_type = "fire_smoke"
        # elif label_list == ["sb_bx"] or label_list == ["sb_dl"] or label_list == ["sb_qx"]:
        #     yolov5_model = yolov5_jmjs
        #     labels = label_list
        #     model_type = "jmjs"
        else:
            yolov5_model = yolov5_rec_defect_x6
            labels_dict = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
            labels = [labels_dict[id] for id in labels_dict]
            if len(label_list) > 0:
                labels = [convert_label(l, "rec_defect") for l in label_list]
            model_type = "rec_defect"
    elif input_data["type"] == "pressplate": 
        yolov5_model = yolov5_ErCiSheBei
        labels = ["kgg_ybh", "kgg_ybf"]
        model_type = "ErCiSheBei"
    elif input_data["type"] == "air_switch":
        yolov5_model = yolov5_ErCiSheBei
        labels = ["kqkg_hz", "kqkg_fz"]
        model_type = "ErCiSheBei"
    elif input_data["type"] == "led":
        yolov5_model = yolov5_ErCiSheBei
        labels = ["zsd_l", "zsd_m"]
        model_type = "ErCiSheBei"
    elif input_data["type"] == "fanpaiqi":
        yolov5_model = yolov5_ErCiSheBei
        model_type = "ErCiSheBei"
        labels = ["fpq_h", "fpq_f", "fpq_jd"]
    elif input_data["type"] == "rotary_switch":
        yolov5_model = yolov5_ErCiSheBei
        labels = ["xnkg_s", "xnkg_zs", "xnkg_ys", "xnkg_z"]
        model_type = "ErCiSheBei"
    elif input_data["type"] == "door":
        yolov5_model = yolov5_ErCiSheBei
        labels = ["xmbhyc", "xmbhzc"]
        model_type = "ErCiSheBei"
    elif input_data["type"] == "key":
        yolov5_model = yolov5_ErCiSheBei
        labels = ["ys"]
        model_type = "ErCiSheBei"
    elif input_data["type"] == "disconnector_texie":
        yolov5_model = yolov5_dztx
        labels = ["he","fen","budaowei"]
        model_type = "disconnector_texie"
    elif input_data["type"] == "person":
        yolov5_model = yolov5_coco
        labels = ["person"]
        model_type = "coco"
    else:
        out_data["msg"] = out_data["msg"] + "Type isn't object; "
        out_data["code"] = 1
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
        out_data["img_result"] = img2base64(img_tag_)
        return out_data

    ## 生成目标检测信息
    if input_data["type"] == "rec_defect":
        if label_list == ["hzyw"] or label_list == ["xdwcr"]:
            cfgs = inference_yolov5(yolov5_model, img_tag, resize=640, pre_labels=labels) # inference
        else:
            cfgs = inference_yolov5(yolov5_model, img_tag, resize=1280, pre_labels=labels, conf_thres=0.7) # inference
    else:
        cfgs = inference_yolov5(yolov5_model, img_tag, resize=640, pre_labels=labels, conf_thres=0.3) # inference
    cfgs = check_iou(cfgs, iou_limit=0.5) # 增加iou机制

    if len(cfgs) == 0: #没有检测到目标
        out_data["msg"] = out_data["msg"] + "; Not find object"
        if input_data["type"] == "rec_defect" or input_data["type"] == "fire_smoke":
            out_data["code"] = 0
        else:
            out_data["code"] = 1
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
        out_data["img_result"] = img2base64(img_tag_)
        return out_data

    ## labels 列表 和 color 列表
    colors = color_list(len(labels))
    color_dict = {}
    name_dict = {}
    for i, label in enumerate(labels):
        color_dict[label] = colors[i]
        if len(status_map) > 0 and label in status_map:
            name_dict[label] = status_map[label]
        elif label in config_object_name.OBJECT_MAP[model_type]:
            name_dict[label] = config_object_name.OBJECT_MAP[model_type][label]
        else:
            name_dict[label] = label

        ## 如果有"real_val"，则输出real_val的值
        if "real_val" in input_data["config"] and isinstance(input_data["config"]["real_val"], str):
            name_dict[label] = input_data["config"]["real_val"]

    ## 画出boxes
    for cfg in cfgs:
        c = cfg["coor"]; label = cfg["label"]
        cv2.rectangle(img_tag_, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), color_dict[label], thickness=2)
        
        if input_data["type"] == "counter" or input_data["type"] == "digital":
            s = int(c[2] - c[0]) # 根据框子大小决定字号和线条粗细。
            img_tag_ = img_chinese(img_tag_, name_dict[label], (c[0], c[1]-s), color=color_dict[label], size=s)
        else:
            s = int((c[2] - c[0]) / 6) # 根据框子大小决定字号和线条粗细。
            img_tag_ = img_chinese(img_tag_, name_dict[label], (c[0], c[1]), color=color_dict[label], size=s)

    ## 求出目标图像的感兴趣区域
    if len(roi) > 0:
        # if len(osd) == 0:
        #     osd = [[0,0,1,0.1],[0,0.9,1,1]]
        # feat_ref = sift_create(img_ref, rm_regs=osd)
        # feat_tag = sift_create(img_tag)
        # M = sift_match(feat_ref, feat_tag, ratio=0.5, ops="Perspective")
        M = cupy_affine(img_ref, img_tag)
        if M is None:
            out_data["msg"] = out_data["msg"] + "; Not enough matches are found"
            roi_tag = roi[0]
        else:
            roi = roi[0]
            coors = [(roi[0],roi[1]), (roi[2],roi[1]), (roi[2],roi[3]), (roi[0],roi[3])]
            coors_ = [list(convert_coor(coor, M)) for coor in coors]
            c_ = np.array(coors_, dtype=int)
            roi_tag = [min(c_[:,0]), min(c_[:, 1]), max(c_[:,0]), max(c_[:,1])]

        ## 画出roi_tag
        c = roi_tag
        cv2.rectangle(img_tag_, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), (255,0,255), thickness=1)
        cv2.putText(img_tag_, "roi", (int(c[0]), int(c[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)

    ## 判断bbox是否在roi中
    bboxes = []
    for cfg in cfgs:
        if len(roi) == 0 or is_include(cfg["coor"], roi_tag, srate=0.5):
            cfg_out = {"label": name_dict[cfg["label"]], "bbox": cfg["coor"], "score": float(cfg["score"])}
            out_data["data"].append(cfg_out)
            bboxes.append(cfg["coor"])

    if len(out_data["data"]) == 0:
        out_data["code"] = 1

    if input_data["type"] == "counter" or input_data["type"] == "digital":
        out_data["data"] = rank_digital(out_data["data"], obj_type=input_data["type"])

    if input_data["type"] == "key":
        out_data["data"] = {"label": input_data["type"], "number": len(bboxes), "boxes": bboxes}
    
    img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
    out_data["img_result"] = img2base64(img_tag_)
    
    return out_data

if __name__ == '__main__':
    from lib_help_base import get_save_head, save_input_data, save_output_data
    json_file = "/data/PatrolAi/result_patrol/led/0109103455_B11-09综保故障指示_input_data.json"
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
    



