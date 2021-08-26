import os
import cv2
import time
import json
import numpy as np
import base64
import argparse
import base64
import json
import requests
import torch
from flask import request,make_response
from flask import Flask, request, jsonify
import lib_image_ops
from lib_inference_yolov5 import load_yolov5_model, inference_yolov5
from lib_analysis_meter import angle_scale, segment2angle, angle2sclae, draw_result
from lib_inference_mrcnn import load_maskrcnn_model, inference_maskrcnn, contour2segment, intersection_arc
from app_disconnector_rec import sift_match, convert_coor


## 加载模型
yolov5_meter = load_yolov5_model("/data/inspection/yolov5/meter.pt") # 加载仪表yolov5模型
maskrcnn_pointer = load_maskrcnn_model("/data/inspection/maskrcnn/pointer.pth") # 加载指针的maskrcnn模型


def conv_coor(coordinates, M, d_ref=(0,0), d_tag=(0,0)):
    """
    根据偏移矩阵M矫正坐标点。
    args:
        coordinates: 参考图上的坐标点，格式如 {"center": [398, 417], "-0.1": [229, 646], "0.9": [641, 593]}
        M: 偏移矩阵，矩阵大小为（2，3） 或（3， 3）
        d_ref: 参考图的平移量，(dx, dy)
        d_tag: 参考图上的偏移量, (dx, dy)
    return:
        coors_tag: 转换后的坐标，格式如 {"center": [398, 417], -0.1: [229, 646], 0.9: [641, 593]}
    """
    ## 将coordinates中的刻度字符串改为浮点型
    coors_float = {}
    for scale in coordinates:
        if scale == "center":
            coors_float["center"] = coordinates["center"]
        else:
            coors_float[float(scale)] = coordinates[scale]

    ## 使用偏移矩阵转换坐标。
    coors_tag = {}
    for scale in coors_float:
        coor = coors_float[scale]
        coor = [coor[0] - d_ref[0], coor[1] - d_ref[1]] # 调整坐标
        coor = convert_coor(coor, M) # 坐标转换
        coor_tag = [coor[0] + d_tag[0], coor[1]+d_tag[1]] # 调整回目标坐标
        coors_tag[scale] = coor_tag
    return coors_tag


def cal_base_angle(coordinates, segment):
    """
    使用角度的方法计算线段的刻度。
    args:
        coordinates: 刻度的坐标点，格式如 {"center": [398, 417], -0.1: [229, 646], 0.9: [641, 593]}
        segment: 线段，格式为 [x1, y1, x2, y2]
    """
    cfg = {}
    for scale in coordinates:
        if scale == "center":
            continue
        angle = segment2angle(coordinates["center"], coordinates[scale])
        cfg[str(angle)] = scale
    if len(cfg) < 2:
        return None
    config = [cfg]
    out_cfg = angle_scale(config)[0]
    seg_ang = segment2angle((segment[0], segment[1]), (segment[2], segment[3]))
    val = round(angle2sclae(out_cfg, seg_ang),3)
    return val


def cal_base_scale(coordinates, segment):
    """
    使用刻度计算指针读数。
    args:
        coordinates: 刻度的坐标点，格式如 {"center": [398, 417], -0.1: [229, 646], 0.9: [641, 593]}
        segment: 线段，格式为 [x1, y1, x2, y2]
    """
    scales = []
    for scale in coordinates:
        if scale == "center":
            continue
        scales.append(scale)
    scales.sort()
    if len(scales) < 2:
        return None
    for i in range(len(scales)-1):
        arc = coordinates["center"] + coordinates[scales[i]] + coordinates[scales[i+1]]
        coor = intersection_arc(segment, arc)
        if coor is not None:
            break
    if coor is None:
        return None
    seg = coordinates["center"] + list(coor)
    scale_1 = scales[i]
    scale_2 = scales[i+1]
    angle_1 = segment2angle(coordinates["center"],coordinates[scale_1])
    angle_2 = segment2angle(coordinates["center"],coordinates[scale_2])
    config = [{str(angle_1): scale_1, str(angle_2): scale_2}]
    out_cfg = angle_scale(config)[0]
    seg_ang = segment2angle((seg[0], seg[1]), (seg[2], seg[3]))
    val = round(angle2sclae(out_cfg, seg_ang),3)
    return val


def inspection_pointer(input_data):

    ## 初始化输入输出信息。
    TIME_START = time.strftime("%m-%d-%H-%M-%S") 
    save_path = os.path.join("pointer", TIME_START)
    os.makedirs(save_path, exist_ok=True)

    out_data = {"code":0, "data":[], "msg": "request sucdess; "} #初始化输出信息

    if input_data["type"] != "pointer":
        out_data["msg"] = out_data["msg"] + "type isn't pointer; "
        return out_data
    
    ## 提取输入请求信息
    img_tag = lib_image_ops.base642img(input_data["image"])
    img_ref = lib_image_ops.base642img(input_data["config"]["img_ref"])
    coordinates_ref = input_data["config"]["coordinates"]

    ## 将输入请求信息可视化
    img_tag_ = img_tag.copy()
    img_ref_ = img_ref.copy()
    cv2.imwrite(os.path.join(save_path, "img_tag.jpg"), img_tag_)
    cv2.imwrite(os.path.join(save_path, "img_ref.jpg"), img_ref_)
    f = open(os.path.join(save_path, "input_data.json"), "w")
    json.dump(input_data, f, ensure_ascii=False)  # 保存输入信息json文件
    f.close()
    for scale in coordinates_ref:  # 将坐标点标注在图片上
        coor = coordinates_ref[scale]
        cv2.circle(img_ref_, (coor[0], coor[1]), 2, (255, 0, 0), 8)
        cv2.putText(img_ref_, str(scale), (int(coor[0])-5, int(coor[1])),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), thickness=2)
    cv2.imwrite(os.path.join(save_path, "img_ref_cfg.jpg"), img_ref_)

    ## 使用yolov5定位参考图和目标图的表盘。
    
    bbox_ref = inference_yolov5(yolov5_meter, img_ref, resize=640)
    if len(bbox_ref) > 0:
        coor_ref = bbox_ref[0]["coor"]
    else:
        coor_ref = [0,0, img_ref.shape[1], img_ref.shape[0]]
    meter_ref = img_ref[int(coor_ref[1]): int(coor_ref[3]), int(coor_ref[0]): int(coor_ref[2])]

    bbox_tag = inference_yolov5(yolov5_meter, img_tag, resize=640)
    if len(bbox_tag) > 0:
        coor_tag = bbox_tag[0]["coor"]
    else:
        coor_tag = [0,0, img_tag.shape[1], img_tag.shape[0]]
    meter_tag = img_tag[int(coor_tag[1]): int(coor_tag[3]), int(coor_tag[0]): int(coor_tag[2])]
    
    # 用maskrcnn检测指针轮廓并且拟合成线段。
    contours, boxes = inference_maskrcnn(maskrcnn_pointer, meter_tag)
    segments = contour2segment(contours, boxes)

    ## 根据与表盘中心的距离更正segment的头尾
    xo = meter_tag.shape[1] / 2; yo = meter_tag.shape[0] / 2
    for i, s in enumerate(segments):
        if (s[0]-xo)**2+(s[1]-yo)**2 > (s[2]-xo)**2+(s[3]-yo)**2:
            segments[i] = [s[2], s[3], s[0], s[1]]

    ## 将segments还原到原图的坐标
    dx = coor_tag[0]; dy = coor_tag[1]
    segments = [[s[0]+dx, s[1]+dy, s[2]+dx, s[3]+dy] for s in segments]

    ## 使用映射变换矫正目标图，并且转换坐标点。
    M = sift_match(meter_ref, meter_tag, ratio=0.5, ops="Perspective")
    coordinates_tag = conv_coor(coordinates_ref, M, coor_ref[:2], coor_tag[:2])

    segment_real = []
    vals = []
    for segment in segments:
        if M is not None:
            val = cal_base_scale(coordinates_tag, segment)
        else:
            val = cal_base_angle(coordinates_tag, segment)
        if val is not None:
            vals.append(val)
            segment_real.append(segment)
    if len(vals) > 0:
        cfg = {"type": "pointer", "values": vals, "segments": segment_real, "bbox": coor_tag}

        out_data["data"] = cfg

    ## 可视化最终计算结果
    cv2.rectangle(img_tag_, (int(coor_tag[0]), int(coor_tag[1])),
                    (int(coor_tag[2]), int(coor_tag[3])), (0, 0, 255), thickness=2)
    cv2.putText(img_tag_, "meter", (int(coor_tag[0])-5, int(coor_tag[1])-5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), thickness=2)
    for seg in segments:
        cv2.line(img_tag_, (int(seg[0]), int(seg[1])), (int(seg[2]), int(seg[3])), (0, 255, 0), 2)
    for i, val in enumerate(vals):
        cv2.putText(img_tag_, str(val), (int(segment_real[i][0])-5, int(segment_real[i][1])),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), thickness=2)
    for scale in coordinates_tag:
        coor = coordinates_tag[scale]
        cv2.circle(img_tag_, (int(coor[0]), int(coor[1])), 2, (255, 0, 0), 8)
        cv2.putText(img_tag_, str(scale), (int(coor[0])-5, int(coor[1])),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), thickness=2)
    cv2.imwrite(os.path.join(save_path, "img_tag_cfg.jpg"), img_tag_)

    for i in range(10):
        torch.cuda.empty_cache()

    return out_data


def main():
    img_ref_file = "images/img_ref.jpg"
    img_tag_file = "/home/yh/app_meter_inference/recognition_resutl/08-24-14-26-06/raw_img.jpg"
    coordinates = {"center": [1113, 476], "-0.1": [890, 779], "0.1": [740, 479], "0.2": [773, 313], "0.3": [887, 178], 
                "0.4": [1050, 101], "0.5": [1230, 106], "0.6": [1390, 191], "0.7": [1488, 349], "0.8": [1502, 531], "0.9": [1423, 701]}
    
    img_tag = lib_image_ops.img2base64(img_tag_file)
    img_ref = lib_image_ops.img2base64(img_ref_file)
    input_data = {"image": img_tag, "config": {"img_ref": img_ref, "coordinates": coordinates}, "type": "pointer"}
    out_data = inspection_pointer(input_data)
    print(out_data)
    # with open(img_ref_file, "rb") as imageFile:
    #     img_1 = imageFile.read()
    # img_2 = base64.b64encode(img_1).decode('utf-8')
    # img_3 = base64.b64decode(str(img_2))
    # img_4 = np.fromstring(img_3, np.uint8)
    # img_5 = cv2.imdecode(img_4, cv2.IMREAD_COLOR)
    # print("a")

if __name__ == '__main__':
    main()
    

