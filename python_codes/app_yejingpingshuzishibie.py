import os
import cv2
import time
import json
import math
from lib_image_ops import base642img, img2base64, img_chinese
# from lib_help_base import oil_high
import numpy as np
import copy

from lib_inference_yolov8 import load_yolov8_model, inference_yolov8
from lib_rcnn_ops import check_iou
from lib_img_registration import roi_registration
from lib_help_base import GetInputData, creat_img_result, draw_region_result, reg_crop,dp_append, img_fill,is_include
from lib_model_import import model_load
from config_model_list import model_threshold_dict

def center_cal(x1,y1,x2,y2):
    return ((x1+x2)/2,(y1+y2)/2)


def dis_cal(center_list):
    dis_list=[]
    for index in range(len(center_list)-1):
        dis=math.sqrt(
                        math.pow((center_list[index+1][0]-center_list[index][0]),2)+
                        math.pow((center_list[index + 1][1] - center_list[index][1]), 2)
                      )
        dis_list.append(dis)
    return dis_list

def dp_self(label_list,dp):
    if len(label_list)<=2:
        dp=len(label_list)-1
        return dp
    coor_list=[item[1] for item in label_list]
    center_list = [center_cal(*coor) for coor in coor_list]
    dis_list = dis_cal(center_list)
    dis_list_re = sorted(dis_list)
    if dis_list_re[-1] / dis_list_re[0] > 1.5:
        dp = len(dis_list_re)-dis_list.index(dis_list_re[-1])
    return dp


def label_withdp(label_list,roi_name,dp_dict,dp):
    label = []
    for item in label_list:
        label.append(str(item[0]))
    if dp_dict == {}:
        if dp ==100:
            dp=dp_self(label_list, dp)
        if dp != 0:
            label = dp_append(label, dp)
        label = "".join(label)
    else:
        dp = int(dp_dict[roi_name])
        if dp ==100:
            dp=dp_self(label_list, dp)
        if dp != 0:
            label = dp_append(label, dp)
        label = "".join(label)
    return label



def inspection_digital_rec(input_data):
    ## 初始化输入输出信息。
    TIME_START = time.strftime("%m%d%H%M%S") + "_"
    if "checkpoint" in input_data and isinstance(input_data["checkpoint"], str) and len(input_data["checkpoint"]) > 0:
        TIME_START = TIME_START + input_data["checkpoint"] + "_"

    out_data = {"code": 0, "type": input_data["type"], "data": {}, "msg": "Success"}  # 初始化输出信息

    ## 提取输入请求信息
    DATA = GetInputData(input_data)
    roi = DATA.roi
    reg_box = DATA.regbox
    img_ref = DATA.img_ref
    img_tag = DATA.img_tag
    dp = DATA.dp
    dp_dict = DATA.dp_dict
    an_type = DATA.type

    ## 将输入请求信息可视化
    img_tag_ = img_tag.copy()
    img_tag_ = img_chinese(img_tag_, TIME_START + an_type, (10, 10), color=(255, 0, 0), size=60)

    if an_type != "digital" and an_type != "counter":
        out_data["msg"] = out_data["msg"] + "type isn't digital or counter; "
        out_data["code"] = 1
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
        out_data["img_result"] = creat_img_result(input_data, img_tag_)  # 返回结果图
        return out_data

    #模型、阈值加载
    yolo_crop, yolo_rec = model_load(an_type)
    conf=model_threshold_dict[an_type]


    # img_ref截取regbox区域用于特征匹配
    if reg_box and len(reg_box) != 0:
        img_ref = reg_crop(img_ref, *reg_box)

    # 使用映射变换矫正目标图，并且转换坐标点。
    roi_tag, _ = roi_registration(img_ref, img_tag, roi)

    ## 将矫正偏移的信息写到图片中
    for name, c in roi_tag.items():
        cv2.rectangle(img_tag_, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (255, 0, 255), thickness=1)
        # cv2.putText(img_tag_, name, (int(c[0]), int(c[1])+10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)
        s = int((c[2] - c[0]) / 10)  # 根据框子大小决定字号和线条粗细。
        img_tag_ = img_chinese(img_tag_, name, (c[0], c[1]), color=(255, 0, 255), size=s)

    # 第一阶段区域识别，截取图像
    bbox_cfg = inference_yolov8(yolo_crop, img_tag,conf_thres=conf)
    # 未检测到目标
    if len(bbox_cfg) < 1:
        coor_list = []
        for roi_tag_item in roi_tag:
            coor_list.append([int(item) for item in roi_tag[roi_tag_item]])
    else:
        coor_list = [item['coor'] for item in bbox_cfg]
    bboxes_list_sort = sorted(coor_list, key=lambda x: x[-1], reverse=False)
    # print("bboxes_list:",bboxes_list)
    roi_name = "no_roi"
    out_data["code"] = 1
    if len(roi_tag) == 0:
        out_data["data"][roi_name] = []
    else:
        for roi_n in roi_tag:
            out_data["data"][roi_n] = []
    for coor in bboxes_list_sort:
        roi_dict = {}
        if len(roi_tag) == 0:
            roi_dict["bbox"] = coor
            mark = True
        else:
            # 去除roi框外，不做识别
            mark = False
            for roi_n in roi_tag:
                if is_include(coor,roi_tag[roi_n],0.5):
                    mark = True
                    roi_dict["bbox"] = coor
                    roi_name = roi_n
                    break
        if not mark:
            continue

        # 640*640填充
        img_empty = img_fill(img_tag, 640, *coor)

        # 二次识别
        bbox_cfg_result = inference_yolov8(yolo_rec, img_empty,conf_thres=conf)
        bbox_cfg_result = check_iou(bbox_cfg_result, 0.2)
        if len(bbox_cfg_result) < 1:
            continue
        # 按横坐标排序组合结果
        label_list = [[item['label'], item['coor']] for item in bbox_cfg_result]
        label_list = sorted(label_list, key=lambda x: x[1][0], reverse=False)

        # 按设置的dp位数添加小数点
        label = label_withdp(label_list,roi_name,dp_dict,dp)
        roi_dict["values"] = label
        roi_dict["score"] = 0.98
        out_data["data"][roi_name].append(roi_dict)
        out_data["code"] = 0
        # value_list.append(label)
        s = (coor[2] - coor[0]) / 50  # 根据框子大小决定字号和线条粗细。
        cv2.putText(img_tag_, str(label), (coor[2], coor[3]), cv2.FONT_HERSHEY_SIMPLEX, round(s), (0, 255, 0),
                    thickness=round(s * 2))
        cv2.rectangle(img_tag_, (int(coor[0]), int(coor[1])), (int(coor[2]), int(coor[3])), (255, 0, 255), thickness=1)

    if out_data["code"] == 1:
        out_data["msg"] = out_data["msg"] + "Can not find digital; "
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
        out_data["img_result"] = creat_img_result(input_data, img_tag_)  # 返回结果图
        return out_data

    ## 主从逻辑中，每个roi框都画一张图
    out_data = draw_region_result(out_data, input_data, roi_tag)

    out_data["msg"] = "Success!"
    # 判断是否位列表式请求
    no_roi = [name.startswith("old_roi") for name in out_data["data"]]
    if all(no_roi):  ## 全为1， 返回True
        value_list = []
        bbox_list = []
        for name, _cfg in out_data["data"].items():
            if len(_cfg) == 0:
                out_data['code'] = 1
                out_data["msg"] = "Can not find digital; "
            else:
                for item in _cfg:
                    value_list.append(item['values'])
                    bbox_list.append(item['bbox'])
        out_data["data"] = {'type': 'digital', "values": value_list, 'bboxes': bbox_list}

    ## 输出可视化结果的图片。
    out_data["img_result"] = creat_img_result(input_data, img_tag_)  # 返回结果图

    return out_data


if __name__ == '__main__':
    img_list = os.listdir('test')
    for item in img_list:
        # if item.endswith(".jpg"):
        #     img_tag_file = "test/"+item
        #     img_tag = img2base64(cv2.imread(img_tag_file))
        #     input_data = {"image": img_tag, "config": {}, "type": "digital"}
        if item.endswith("input_data.json"):
            with open("test/" + item, "r", encoding="utf8") as f:
                input_data = json.load(f)
            # print(input_data)
            out_data = inspection_digital_rec(input_data)
            print(item, out_data['msg'])
            # print(out_data['data']['values'])
            print("==================================")
