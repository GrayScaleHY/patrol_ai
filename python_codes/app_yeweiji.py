import os
import cv2
import time
import json
import math
from lib_image_ops import base642img, img2base64, img_chinese
import numpy as np
from lib_sift_match import sift_match, convert_coor, fft_registration
from lib_help_base import GetInputData,color_list,is_include,save_output_data,get_save_head,save_output_data
from lib_inference_yolov5 import inference_yolov5,check_iou
import config_object_name
from lib_inference_yolov5 import load_yolov5_model

yolov5_yeweiji = load_yolov5_model("/data/home/zgl/datasets/Project/runs/train/ywj6/weights/best.pt") # 加载液位计模型

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

## 计算油位
def cal_oil_value(coordinates,oil_bbox,oil_type="updown"):
    """
        使用刻度计算液位高度。
        args:
            coordinates: 刻度的坐标点，格式如 {"0": [300, 600],....,"1": [600, 200]} 至少包含最小、最大两个刻度
            oil_bbox: 检测出的液位框，格式为[x0, y0, x1, y1] 
    """
    if len(coordinates) < 2:
        return None

    if oil_type == "updown":            #默认竖向
        for k,v in coordinates.items(): #取出y值
            coordinates[k] = v[1]   #{'0': 600, '10': 200, '20': 50} 刻度值越大y坐标越小

        ##找出coordinates中距离液位最近的前后两个值
    oil_h = min(oil_bbox[1],oil_bbox[3]) #返回ymin 对应液位最高点
    y_list = list(coordinates.values())
    up_list = []
    down_list = []
    for j in y_list:
        if j >= oil_h:
            down_list.append(j)
        else:
            up_list.append(j)

    index_down = list(coordinates.values()).index(min(down_list))
    index_up = list(coordinates.values()).index(max(up_list))

    _down = list(coordinates.keys())[index_down] #接近检测值的下刻度
    _up = list(coordinates.keys())[index_up]     #接近检测值的上刻度
    
    #返回s对应的刻度值
    if oil_type == "updown":        ## 默认竖向
        y_min = max(up_list)  # 对应 上刻度的高度 ymin
        y_max = min(down_list)  # 对应 下刻度的高度 ymax
        rate_ = (y_max - oil_h)/(y_max - y_min)
        value =  float(_down) + rate_ * (float(_up) - float(_down))  # 刻度下线 + 刻度(上-下)* 百分比 # 刻度下线 + 刻度(上-下)* 百分比
    else:
        # oil_type == "liftright":
        pass
    return abs(value)

def inspection_level_gauge(input_data):

    ## 提取输入请求信息
    DATA = GetInputData(input_data)
    img_tag = DATA.img_tag
    img_ref = DATA.img_ref
    roi = DATA.roi
    osd = DATA.osd
    pointers = DATA.pointers #{"0": [1500, 2100],"1":[1680, 780]}
    dp = DATA.dp
    checkpoint = DATA.checkpoint
    an_type = DATA.type
    status_map = DATA.status_map

    ## 初始化输出结果
    out_data = {"code":0, "data":{}, "img_result": DATA.img_tag, "msg": "Request " + an_type + ";"}

    ## 画上点位名称和osd区域
    img_tag_ = img_tag.copy()
    img_tag_ = img_chinese(img_tag_, checkpoint + an_type , (10, 10), color=(255, 0, 0), size=60)
    
    for o_ in osd:  ## 如果配置了感兴趣区域，则画出osd区域
        cv2.rectangle(img_tag_, (int(o_[0]), int(o_[1])),(int(o_[2]), int(o_[3])), (255, 0, 255), thickness=1)
        cv2.putText(img_tag_, "osd", (int(o_[0]), int(o_[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)

    if input_data["type"] != "level_gauge":
        out_data["msg"] = out_data["msg"] + "type isn't level_gauge; "
        out_data["code"] = 1
    else:
        out_data["code"] = 0
        yolov5_model = yolov5_yeweiji
        labels_dict = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
        labels = [labels_dict[id] for id in labels_dict]
        model_type = "level_gauge"

        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
        out_data["img_result"] = img2base64(img_tag_)
        
    ## 用yolov5检测油位
    cfgs = inference_yolov5(yolov5_yeweiji, img_tag, resize=640, pre_labels=labels) 
    cfgs = check_iou(cfgs, iou_limit=0.5)
    # print("cfgs:", cfgs)

    if len(cfgs) == 0: #没有检测到目标
        out_data["msg"] = out_data["msg"] + "; Not find object"
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

    ## 画出油位上部液位线 
    for cfg in cfgs:
        c = cfg["coor"]; label = cfg["label"]
        cv2.line(img_tag_,(int(c[0]),int(c[1])),(int(c[2]),int(c[1])), color_dict[label], thickness=2) #(x0,y0),(x1,y0)
        s = int((c[2] - c[0]) / 6) # 根据框子大小决定字号和线条粗细。
        img_tag_ = img_chinese(img_tag_, name_dict[label], (c[0], c[1]), color=color_dict[label], size=s)

# ## 求出目标图像的感兴趣区域
    if len(roi)!=0 and img_ref is not None:
        # if len(osd) == 0:
        #     osd = [[0,0,1,0.1],[0,0.9,1,1]]
        # feat_ref = sift_create(img_ref, rm_regs=osd)
        # feat_tag = sift_create(img_tag)
        # M = sift_match(feat_ref, feat_tag, ratio=0.5, ops="Perspective")
        M = fft_registration(img_ref, img_tag)
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

    ## 使用映射变换矫正目标图，并且转换坐标点。
    pointers_tag = conv_coor(pointers, M)
    for scale in pointers_tag:
        coor = pointers_tag[scale]
        cv2.circle(img_tag_, (int(coor[0]), int(coor[1])), 1, (255, 0, 255), 8)
        cv2.putText(img_tag_, str(scale), (int(coor[0]), int(coor[1])),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), thickness=1)
       
## 判断bbox是否在roi中
    bboxes = []
    for cfg in cfgs:
        if len(roi) == 0 or is_include(cfg["coor"], roi_tag, srate=0.5):
            cfg_out = {"label": name_dict[cfg["label"]], "bbox": cfg["coor"], "score": float(cfg["score"])}
            out_data["data"] = cfg_out
            bboxes.append(cfg["coor"])
    if len(out_data["data"]) == 0:
        out_data["code"] = 1
    
    value = cal_oil_value(pointers,cfgs[0]["coor"],oil_type="updown")
    value = round(value, dp)
    out_data["data"]["value"] = value

    ## 可视化最终计算结果
    s = (c[2] - c[0]) / 30 # 根据框子大小决定字号和线条粗细。
    cv2.putText(img_tag_, str(value), (int(c[2]),int(c[1])),cv2.FONT_HERSHEY_SIMPLEX, round(s), (0, 255, 0), thickness=round(s))
    out_data["img_result"] = img2base64(img_tag_)
    
    return out_data

if __name__ == '__main__':

    from lib_help_base import get_save_head, save_input_data, save_output_data
    img_tag_file = "/data/home/zgl/datasets/ywj_partitioned/test/20230106165229_5.jpg"
    img_tag = img2base64(cv2.imread(img_tag_file))
    input_data = {"image": img_tag, 
    "config": {
        "img_ref": img_tag, 
        # "pointers":{"-30": [0.56, 0.67],"0":[0.56, 0.59],
        # "70":[0.53, 0.37],"80":[0.54, 0.34],"90":[0.57, 0.313]},
        # "bboxes":{"roi":[0.5,0.7,0.56,0.25]},
        "pointers":{"20": [0.54, 0.66],"100":[0.54, 0.565],
        "160":[0.54, 0.495]},
        "bboxes":{"roi":[0.47,0.21,0.57,0.75]},
    },
        "type": "level_gauge"}
    out_data = inspection_level_gauge(input_data)
    save_dir, name_head = get_save_head(input_data)
    save_input_data(input_data, save_dir, name_head, draw_img=True)
    save_output_data(out_data, save_dir, name_head)

    print("-----------------------------------------------")
    for s in out_data:
        if s != "img_result":
            print(s,":",out_data[s])
    print("----------------------------------------------")
