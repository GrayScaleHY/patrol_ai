'''
静默监视
安全帽 工服 吸烟 倒地
'''
from flask import Flask, request, jsonify
import os
import time
import cv2
import json
import glob
import argparse
import numpy as np
import base64
from lib_help_base import color_list
from lib_image_ops import base642img, img2base64, img_chinese
from config_load_models_var import yolov5_rec_defect,yolov5_action
from lib_inference_yolov5 import inference_yolov5, check_iou

def get_input_data(input_data):
    """
    提取input_data中的信息。
    return:
        img_tag: 目标图片数据
    """
    img_tag = base642img(input_data["image"])

    return img_tag

def inspection_silent_monitor(input_data):
    """
    yolov5的目标检测推理。
    """
    #  将输入请求信息可视化
    TIME_START = time.strftime("%m%d%H%M%S") + "_"
    if "checkpoint" in input_data and isinstance(input_data["checkpoint"], str) and len(input_data["checkpoint"]) > 0:
        TIME_START = TIME_START + input_data["checkpoint"] + "_"
    save_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_path = os.path.join(save_path, "result_patrol", input_data["type"])
    os.makedirs(save_path, exist_ok=True)
    f = open(os.path.join(save_path, TIME_START + "input_data.json"), "w")
    json.dump(input_data, f, ensure_ascii=False)  # 保存输入信息json文件
    f.close()

    #  初始化输入输出信息。
    img_tag = get_input_data(input_data)
    out_data = {"code": 1, "data": [], "img_result": input_data["image"],
                "msg": "Success request object detect; "}  # 初始化out_data
    img_tag_ = img_tag.copy()
    cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag.jpg"), img_tag)  # 将输入图片可视化


    #  模型推理
    #  缺陷 安全帽 工服 吸烟
    yolov5_model = yolov5_rec_defect
    labels_dict = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
    labels_rec_defect = [labels_dict[id] for id in labels_dict]
    cfgs = inference_yolov5(yolov5_model, img_tag, resize=640, pre_labels=labels_rec_defect)  # inference
    cfgs_rec_defect = check_iou(cfgs, iou_limit=0.5)  # 增加iou机制


    #  倒地
    yolov5_model = yolov5_action
    labels_dict = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
    labels_action = [labels_dict[id] for id in labels_dict]
    cfgs = inference_yolov5(yolov5_model, img_tag, resize=640, pre_labels=labels_action)  # inference
    cfgs_action = check_iou(cfgs, iou_limit=0.5)  # 增加iou机制


    cfgs = cfgs_rec_defect + cfgs_action
    labels = labels_rec_defect + labels_action
    #  筛选4种需要的类别
    select_labels = ['wcaqm', 'wcgz', 'xy', '摔倒']
    res_cfgs = []
    for i in cfgs:
        # print(i)
        if i['label'] in select_labels:
            res_cfgs.append(i)

    cfgs = res_cfgs

    ## labels 列表 和 color 列表
    colors = color_list(len(labels))
    color_dict = {}
    name_dict = {}
    for i, label in enumerate(labels):
        color_dict[label] = colors[i]
        name_dict[label] = label

    # print(color_dict, name_dict)


    if len(cfgs) == 0:  # 没有检测到目标
        out_data["msg"] = out_data["msg"] + "; Not find object"
        out_data["code"] = 0
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
        out_data["img_result"] = img2base64(img_tag_)
        cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag_cfg.jpg"), img_tag_)
        return out_data

    else:
        ## 画出boxes
        for cfg in cfgs:
            c = cfg["coor"];
            label = cfg["label"]
            cv2.rectangle(img_tag_, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), color_dict[label], thickness=2)

            s = int((c[2] - c[0]) / 6)  # 根据框子大小决定字号和线条粗细。
            img_tag_ = img_chinese(img_tag_, name_dict[label], (c[0], c[1]), color=color_dict[label], size=s)

    ## 判断bbox是否在roi中
    bboxes = []
    for cfg in cfgs:
        cfg_out = {"label": name_dict[cfg["label"]], "bbox": cfg["coor"], "score": float(cfg["score"])}
        out_data["data"].append(cfg_out)
        bboxes.append(cfg["coor"])

    ## 可视化计算结果
    f = open(os.path.join(save_path, TIME_START + "out_data.json"), "w")
    json.dump(out_data, f, ensure_ascii=False, indent=2)  # 保存输入信息json文件
    f.close()

    ## 输出可视化结果的图片。
    img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
    out_data["img_result"] = img2base64(img_tag_)
    # cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag_cfg.jpg"), img_tag_)
    cv2.imwrite(os.path.join(TIME_START + "img_tag_cfg.jpg"), img_tag_)

    return out_data

if __name__ == '__main__':
    json_file = "test.json"
    f = open(json_file,"r",encoding='utf-8')
    input_data = json.load(f)
    f.close()
    out_data = inspection_silent_monitor(input_data)
    print("inspection_object_detection result:")
    print("-----------------------------------------------")
    for s in out_data:
        if s != "img_result":
            print(s,":",out_data[s])
    print("----------------------------------------------")
