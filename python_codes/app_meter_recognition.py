"""
仪表识别app，分析仪表中的信息。
接口详情在https://git.utapp.cn/xunshi-ai/json-http-interface/-/wikis/JSON-HTTP%E6%8E%A5%E5%8F%A3
input_data: {"images": img_base64, "config": [{"角度": 刻度, "角度": 刻度, ..}, ..], "type": ["pointer", "counter", "meter"]}
out_data: {"code": 0}
"""


import math
import numpy as np
import cv2
import lib_inference_mrcnn
from lib_inference_yolov5 import load_yolov5_model, inference_yolov5
from lib_analysis_meter import angle_scale, segment2angle, angle2sclae, draw_result
from lib_inference_mrcnn import load_maskrcnn_model, inference_maskrcnn, contour2segment
import base64
import time
import os
from lib_image_ops import base642img
import json


yolov5_meter = load_yolov5_model("yolov5/saved_model/best_meter.pt") # 加载仪表yolov5模型
yolov5_counter= load_yolov5_model("yolov5/saved_model/best_digital.pt") # 加载记数yolov5模型
maskrcnn_pointer = load_maskrcnn_model("detectron2/saved_model/best_pointer.pth") # 加载指针的maskrcnn模型

global TIME_START
TIME_START = time.strftime("%m-%d-%H-%M-%S")
print(TIME_START, ": app_meter_recognition.py start")
print("=================================================")


def meter_loc(input_data):
    """
    仪表定位。
    """

    out_data = {"code": 0, "data":[], "msg": "Success request meter"} # 初始化out_data
    cfg = {"type": "meter", "bboxes":[]}

    img_base64 = input_data["image"]
    img = base642img(img_base64) # base64格式转numpy格式

    bbox_meters = inference_yolov5(yolov5_meter, img, resize=640) # yolov5模型识别
    
    for bbox in bbox_meters:
        if bbox["label"] == 'meter':
            cfg["bboxes"].append(bbox["coor"])
        else:
            print("warning: label is not meter !")

    if len(bbox_meters) == 0:
        out_data["msg"] = out_data["msg"] + "; Not find meter"
    else:
        out_data["data"].append(cfg)

    save_image = os.path.join("recognition_resutl",TIME_START, "raw_img_meter.jpg")
    draw_result(input_data, out_data, save_image)
    print("result draw in",save_image)
    print("============================================")
    
    return out_data


def rec_counter(input_data):
    """
    动作次数数字识别。
    """

    out_data = {"code": 0, "data":[], "msg": "Success request counter"} # 初始化out_data

    img_base64 = input_data["image"]
    img = base642img(img_base64) # base64格式转numpy格式

    ## 表盘坐标
    bbox_meters = inference_yolov5(yolov5_meter, img, resize=640)
    if len(bbox_meters) == 0:
        meter_coor = [0.0, 0.0, float(img.shape[1]), float(img.shape[0])]
        bbox_meters = [{'label': 'meter', 'coor': meter_coor, "score": 0}]

    
    for bbox_meter in bbox_meters:

        ## 表盘信息
        meter_coor = bbox_meter['coor']
        m_x = meter_coor[0]; m_y = meter_coor[1]
        img_meter = img[int(meter_coor[1]): int(meter_coor[3]), int(meter_coor[0]): int(meter_coor[2])]

        bbox_counters = inference_yolov5(yolov5_counter, img_meter, resize=640)

        if len(bbox_counters) == 0:
            continue
        
        ## 根据从左到右的规则对bbox_digitals的存放排序
        l = [a['coor'][0] for a in bbox_counters]
        rank = [index for index,value in sorted(list(enumerate(l)),key=lambda x:x[1])]

        ## 将vals和bboxes添加进out_data
        vals = []; bboxes = []
        for i in rank:
            vals.append(int(bbox_counters[i]['label']))
            coor = bbox_counters[i]['coor']
            coor_2 = [coor[0]+m_x, coor[1]+m_y, coor[2]+m_x, coor[3]+m_y]
            bboxes.append(coor_2)

        if len(vals) > 0:
            out = {"type": "counter", "values": vals, "bboxes": bboxes}
            out_data['data'].append(out)

    if len(out_data["data"]) == 0:
        out_data["msg"] = out_data["msg"] + ";Not find counter !"

    save_image = os.path.join("recognition_resutl",TIME_START, "raw_img_counter.jpg")
    draw_result(input_data, out_data, save_image)
    print("result draw in",save_image)

    return out_data


def rec_pointer(input_data):
    """
    分析指针类仪表。
    """

    out_data = {"code": 0, "data":[], "msg": "Success request pointer"} # 初始化out_data

    img_base64 = input_data["image"]
    img = base642img(img_base64) # base64格式转numpy格式

    bbox_meters = inference_yolov5(yolov5_meter, img, resize=640) # yolov5模型识别
    
    if len(bbox_meters) == 0:
        meter_coor = [0.0, 0.0, float(img.shape[1]), float(img.shape[0])]
        bbox_meters = [{'label': 'meter', 'coor': meter_coor, "score": 0}]

    ## 对bbox_meters进行排序
    meter_coors = [bbox["coor"] for bbox in bbox_meters]
    meter_coors = np.array(meter_coors)
    meter_coors = meter_coors[np.argsort(meter_coors[:,0]),:]
    meter_coors = meter_coors.tolist()

    ## 将out_cfgs补到和bbox_meters一样多。
    configs = input_data["config"]
    if len(bbox_meters) > len(configs):
        configs = configs + [configs[-1]] * (len(bbox_meters) - len(configs))
    out_cfgs = angle_scale(input_data["config"])

    segments_all = []
    for i, meter_coor in enumerate(meter_coors):

        ## 表盘信息
        m_x = meter_coor[0]; m_y = meter_coor[1] # 表盘起始坐标
        img_meter = img[int(meter_coor[1]): int(meter_coor[3]), int(meter_coor[0]): int(meter_coor[2])]

        ## 刻度信息
        out_cfg = out_cfgs[i]
        
        ## 用maskrcnn检测指针轮廓并且拟合成线段。
        contours, boxes = inference_maskrcnn(maskrcnn_pointer, img_meter)
        segments = contour2segment(contours, boxes)

        vals = []
        segs = []
        for segment in segments:
            ## 重新调整线段的头尾
            x0 = [segment[0], segment[1]]; x1 = [segment[2], segment[3]]
            y = [img_meter.shape[1] / 2 , img_meter.shape[0] / 2]
            l0 = math.sqrt(sum([(a - b)**2 for (a,b) in zip(x0, y)]))
            l1 = math.sqrt(sum([(a - b)**2 for (a,b) in zip(x1, y)]))
            if l0 > l1:
                seg = [segment[2], segment[3], segment[0], segment[1]]
            else:
                seg = segment
            ## 将线段坐标重新映射回原图
            segment = [seg[0]+m_x, seg[1]+m_y, seg[2]+m_x, seg[3]+m_y]
            seg_ang = segment2angle((segment[0], segment[1]), (segment[2], segment[3]))
            val = angle2sclae(out_cfg, seg_ang)
            vals.append(round(val,3))
            segs.append(segment)

        if len(segs) > 0:
            out_data["data"].append({"type": "pointer", "values": vals, "segments": segs, "bbox": meter_coor})
            
    if len(out_data["data"]) == 0:
        out_data["msg"] = out_data["msg"] + ";Not find pointer"

    save_image = os.path.join("recognition_resutl",TIME_START, "raw_img_pointer.jpg")
    draw_result(input_data, out_data, save_image)
    print("result draw in",save_image)

    return out_data


def meter_rec(input_data):
    """
    仪表识别
    """

    global TIME_START
    TIME_START = time.strftime("%m-%d-%H-%M-%S")

    save_image = os.path.join("recognition_resutl",TIME_START, "raw_img_meter_rec.jpg")

    out_data_ = meter_loc(input_data)

    out_data = {"code": 0, "data":[], "msg": "Success request"} # 初始化out_data
    for type_ in input_data["type"]:
        if type_ == "counter":
            out_data_ = rec_counter(input_data)
    # out_data_1 = rec_counter(input_data)
        elif type_ == "pointer":
            out_data_ = rec_pointer(input_data)
    # out_data_2 = rec_pointer(input_data)
        else:
            out_data["msg"] == "type of input data is not standard !"
            continue
    
        out_data["msg"] = out_data["msg"] + ";" + out_data_["msg"]
        out_data["data"] = out_data["data"] + out_data_["data"]

    draw_result(input_data, out_data, save_image)
    print("result draw in",save_image)
    print(out_data)
    print("====================================================")
    
    return out_data


if __name__ == '__main__':
    input_img = '/home/yh/meter_recognition/test/1_155614.jpg'
    # img_raw = cv2.imread(img_file)
    with open(input_img, "rb") as imageFile:
        img = imageFile.read()
    image = base64.b64encode(img).decode('utf-8')
    data = {"image": image, "config": [{"228.17": 0,"307.97": 150}], "type": ["pointer", "counter"]}
    # data = {"image": image, "config":[{"185.57": -30,"338.95": 50}, {"183.60": 0,"340.56": 100}], "type": ["counter", "pointer"]}
    # out_data = meter_loc(data)
    out_data = meter_rec(data)
    print(out_data)
    