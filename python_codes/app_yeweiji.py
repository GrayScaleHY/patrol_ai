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

def get_input_data(input_data):
    """
    提取input_data中的信息。
    return:
        img_tag: 目标图片数据
        img_ref: 模板图片数据
        roi: 感兴趣区域, 结构为[xmin, ymin, xmax, ymax]
        sp: 液位计的形状，圆形或方形
        dp: 需要保留的小数位
    """
    img_tag = base642img(input_data["image"])
    

    ## 是否有模板图
    img_ref = None
    if "img_ref" in input_data["config"]:
        if isinstance(input_data["config"]["img_ref"], str):
            img_ref = base642img(input_data["config"]["img_ref"])  

    ## 感兴趣区域
    roi = None # 初始假设
    if "bboxes" in input_data["config"]:
        if isinstance(input_data["config"]["bboxes"], dict):
            if "roi" in input_data["config"]["bboxes"]:
                if isinstance(input_data["config"]["bboxes"]["roi"], list):
                    if isinstance(input_data["config"]["bboxes"]["roi"][0], list):
                        roi = input_data["config"]["bboxes"]["roi"][0]
                    else:
                        roi = input_data["config"]["bboxes"]["roi"]
                    W = img_ref.shape[1]; H = img_ref.shape[0]
                    roi = [int(roi[0]*W), int(roi[1]*H), int(roi[2]*W), int(roi[3]*H)]  
    
    ## 其他信息
    sp = 1
    if "sp" in input_data["config"]:
        if isinstance(input_data["config"]["sp"], int):
            if input_data["config"]["sp"] != -1:
                sp = input_data["config"]["sp"]
    dp = 3
    if "dp" in input_data["config"]:
        if isinstance(input_data["config"]["dp"], int):
            if input_data["config"]["dp"] != -1:
                dp = input_data["config"]["dp"]
    
    return img_tag, img_ref, roi, sp, dp


def inspection_level_gauge(input_data):

    ## 初始化输入输出信息。
    TIME_START = time.strftime("%m%d%H%M%S") + "_"
    if "checkpoint" in input_data and isinstance(input_data["checkpoint"], str) and len(input_data["checkpoint"]) > 0:
        TIME_START = TIME_START + input_data["checkpoint"] + "_"
    save_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_path = os.path.join(save_path, "result_patrol", input_data["type"])
    os.makedirs(save_path, exist_ok=True)
    f = open(os.path.join(save_path, TIME_START + "input_data.json"), "w")
    json.dump(input_data, f, ensure_ascii=False)  # 保存输入信息json文件
    f.close()

    out_data = {"code":0, "data":{}, "img_result": "image", "msg": "request sucdess; "} #初始化输出信息
    
    ## 提取输入请求信息
    img_tag, img_ref, roi, sp, dp= get_input_data(input_data)

    ## 将输入请求信息可视化
    img_tag_ = img_tag.copy()
    img_tag_ = img_chinese(img_tag_, TIME_START + input_data["type"] , (10, 10), color=(255, 0, 0), size=60)
    

    if input_data["type"] != "level_gauge":
        out_data["msg"] = out_data["msg"] + "type isn't level_gauge; "
        out_data["code"] = 1
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
        out_data["img_result"] = img2base64(img_tag_)
        cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag_cfg.jpg"), img_tag_)
        return out_data

    cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag.jpg"), img_tag)
    if roi is not None and img_ref is not None:   ## 如果配置了感兴趣区域，则画出感兴趣区域
        img_ref_ = img_ref.copy()
        cv2.imwrite(os.path.join(save_path, TIME_START + "img_ref.jpg"), img_ref_)
        cv2.rectangle(img_ref_, (int(roi[0]), int(roi[1])),(int(roi[2]), int(roi[3])), (255, 0, 255), thickness=1)
        cv2.putText(img_ref_, "roi", (int(roi[0]), int(roi[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)
        cv2.imwrite(os.path.join(save_path, TIME_START + "img_ref_cfg.jpg"), img_ref_)

    ## 如果没有配置roi，则自动识别表盘作为roi
    if roi is None:
        M = None
    else: 
        feat_ref = sift_create(img_ref, rm_regs=[[0,0,1,0.1],[0,0.9,1,1]])
        feat_tag = sift_create(img_tag, rm_regs=[[0,0,1,0.1],[0,0.9,1,1]])
        M = sift_match(feat_ref, feat_tag, ratio=0.5, ops="Perspective")
    
    if M is None:
        roi_tag = [0,0, img_tag.shape[1], img_tag.shape[0]]
    else:
        coors = [(roi[0],roi[1]), (roi[2],roi[1]), (roi[2],roi[3]), (roi[0],roi[3])]
        coors_ = []
        for coor in coors:
            coors_.append(list(convert_coor(coor, M)))
        xs = [coor[0] for coor in coors_]
        ys = [coor[1] for coor in coors_]
        xmin = max(0, min(xs)); ymin = max(0, min(ys))
        xmax = min(img_tag.shape[1], max(xs)); ymax = min(img_tag.shape[0], max(ys))
        roi_tag = [xmin, ymin, xmax, ymax]
    img_roi = img_tag[int(roi_tag[1]): int(roi_tag[3]), int(roi_tag[0]): int(roi_tag[2])]

    ## 使用映射变换矫正目标图，并且转换坐标点。

    ## 将矫正偏移的信息写到图片中
    cv2.rectangle(img_tag_, (int(roi_tag[0]), int(roi_tag[1])),(int(roi_tag[2]), int(roi_tag[3])), (255, 0, 255), thickness=1)
    cv2.putText(img_tag_, "roi", (int(roi_tag[0]), int(roi_tag[1]-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=1)     

    # 用maskrcnn检测油和空气的mask.
    contours, _, (masks, classes, scores) = inference_maskrcnn(maskrcnn_oil, img_roi)
    if len(masks) < 1:
        out_data["msg"] = out_data["msg"] + "Can not find oil_lelvel; "
        out_data["code"] = 1
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
        out_data["img_result"] = img2base64(img_tag_)
        cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag_cfg.jpg"), img_tag_)
        return out_data

    ## 计算油率
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
    cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag_cfg.jpg"), img_tag_)

    ## 输出可视化结果的图片。
    f = open(os.path.join(save_path, TIME_START + "output_data.json"), "w", encoding='utf-8')
    json.dump(out_data, f, indent=2, ensure_ascii=False)
    f.close()
    out_data["img_result"] = img2base64(img_tag_)

    return out_data


if __name__ == '__main__':
    img_tag_file = "test/#0755_org_0.jpg"
    img_tag = img2base64(cv2.imread(img_tag_file))
    input_data = {"image": img_tag, "config": {}, "type": "level_gauge"}
    out_data = inspection_level_gauge(input_data)
    

