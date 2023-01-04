import os
import cv2
import time
import json
import math
from lib_image_ops import base642img, img2base64, img_chinese
from lib_help_base import oil_high
import numpy as np
from lib_inference_mrcnn import load_maskrcnn_model, inference_maskrcnn
from lib_sift_match import sift_match, convert_coor, sift_create
from config_load_models_var import maskrcnn_oil
from lib_help_base import GetInputData

def inspection_level_gauge(input_data):

    ## 提取输入请求信息
    DATA = GetInputData(input_data)
    checkpoint = DATA.checkpoint; an_type = DATA.type
    img_tag = DATA.img_tag; img_ref = DATA.img_ref
    roi = DATA.roi; osd = DATA.osd; dp = DATA.dp
    
    ## 初始化输出结果
    out_data = {"code":0, "data":{}, "img_result": input_data["image"], "msg": "Request " + an_type + ";"} #初始化输出信息

    ## 画上点位名称和osd区域
    img_tag_ = img_tag.copy()
    img_tag_ = img_chinese(img_tag_, checkpoint + an_type , (10, 10), color=(255, 0, 0), size=60)
    for o_ in osd:  ## 如果配置了感兴趣区域，则画出osd区域
        cv2.rectangle(img_tag_, (int(o_[0]), int(o_[1])),(int(o_[2]), int(o_[3])), (255, 0, 255), thickness=1)
        cv2.putText(img_tag_, "osd", (int(o_[0]), int(o_[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)

    if input_data["type"] != "level_gauge":
        out_data["msg"] = out_data["msg"] + "type isn't level_gauge; "
        out_data["code"] = 1
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
        out_data["img_result"] = img2base64(img_tag_)
        return out_data

    ## 求出目标图像的感兴趣区域
    if len(roi) > 0:
        if len(osd) > 0:
            osd = [[0,0,1,0.1],[0,0.9,1,1]]
        feat_ref = sift_create(img_ref, rm_regs=osd)
        feat_tag = sift_create(img_tag)
        M = sift_match(feat_ref, feat_tag, ratio=0.5, ops="Perspective")
        if M is None:
            out_data["msg"] = out_data["msg"] + "; Not enough matches are found"
            roi_tag = roi[0]
        else:
            roi = roi[0]
            coors = [(roi[0],roi[1]), (roi[2],roi[1]), (roi[2],roi[3]), (roi[0],roi[3])]
            coors_ = [list(convert_coor(coor, M)) for coor in coors]
            c_ = np.array(coors_, dtype=int)
            roi_tag = [min(c_[:,0]), min(c_[:, 1]), max(c_[:,0]), max(c_[:,1])]
    else:
        roi_tag = [0,0, img_tag.shape[1], img_tag.shape[0]]
    img_roi = img_tag[int(roi_tag[1]): int(roi_tag[3]), int(roi_tag[0]): int(roi_tag[2])]
    
    ## 画出roi_tag
    c = roi_tag
    cv2.rectangle(img_tag_, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), (255,0,0), thickness=1)
    cv2.putText(img_tag_, "roi", (int(c[0]), int(c[1]) + 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)

    # 用maskrcnn检测油和空气的mask.
    contours, _, (masks, classes, scores) = inference_maskrcnn(maskrcnn_oil, img_roi)
    if len(masks) < 1:
        out_data["msg"] = out_data["msg"] + "Can not find oil_lelvel; "
        out_data["code"] = 1
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
        out_data["img_result"] = img2base64(img_tag_)
        return out_data

    ## 计算油率
    sp = 1
    air_s = 0; oil_s = 0
    if 0 in classes:
        air_s = np.sum(masks[list(classes).index(0)])
    if 1 in classes:
        oil_s= np.sum(masks[list(classes).index(1)])
    if sp:
        value = oil_s / (air_s + oil_s)
    else:
        oil_h = oil_high(oil_s, air_s + oil_s)
        R = math.sqrt((air_s + oil_s) / math.pi) #根据圆面积求圆半径
        value = oil_h / (R * 2)
    value = round(value, dp)

    ## 将contours还原到原图的坐标
    dx = roi_tag[0]; dy = roi_tag[1]
    for i in range(len(contours)):
        contours[i][:,0,0] = contours[i][:,0,0] + dx
        contours[i][:,0,1] = contours[i][:,0,1] + dy

    out_data["data"] = {"label": "oil_rate", "value": value}
    
    ## 可视化最终计算结果
    s = (roi_tag[2] - roi_tag[0]) / 400 # 根据框子大小决定字号和线条粗细。
    cv2.putText(img_tag_, str(value), (0+round(s)*10, 0+round(s)*30), cv2.FONT_HERSHEY_SIMPLEX, round(s), (0, 255, 0), thickness=round(s*2))
    cv2.drawContours(img_tag_,contours,-1,(0,0,255),1)
    out_data["img_result"] = img2base64(img_tag_)

    return out_data

if __name__ == '__main__':
    img_tag_file = "test/#0755_org_0.jpg"
    img_tag = img2base64(cv2.imread(img_tag_file))
    input_data = {"image": img_tag, "config": {}, "type": "level_gauge"}
    out_data = inspection_level_gauge(input_data)
    

