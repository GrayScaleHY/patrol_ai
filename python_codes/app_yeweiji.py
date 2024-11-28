import os
import cv2
import time
import json
import math
from copy import deepcopy
from lib_image_ops import img2base64, img_chinese
import numpy as np
from lib_img_registration import roi_registration, convert_coor
from lib_help_base import GetInputData,color_list, is_include, save_output_data, get_save_head, save_output_data, creat_img_result
from lib_inference_yolov8 import load_yolov8_model, inference_yolov8
from lib_rcnn_ops import check_iou
from app_yejingpingshuzishibie import yolov8_jishukuang,yolov8_jishushibie,img_fill
import config_object_name

yolov8_yeweiji = load_yolov8_model("/data/PatrolAi/yolov8/yeweiji.pt") # 加载液位计模型


def conv_coor(coordinates, M, d_ref=(0, 0), d_tag=(0, 0)):
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
    # 将coordinates中的刻度字符串改为浮点型
    coors_float = {}
    for scale in coordinates:
        if scale == "center":
            coors_float["center"] = coordinates["center"]
        else:
            coors_float[float(scale)] = coordinates[scale]

    if M is None:
        return coors_float

    # 使用偏移矩阵转换坐标。
    coors_tag = {}
    for scale in coors_float:
        coor = coors_float[scale]
        coor = [coor[0] - d_ref[0], coor[1] - d_ref[1]] # 调整坐标
        coor = convert_coor(coor, M) # 坐标转换
        coor_tag = [coor[0] + d_tag[0], coor[1]+d_tag[1]] # 调整回目标坐标
        coors_tag[scale] = coor_tag
    return coors_tag


# 计算油位
def cal_oil_value(coordinates, oil_bbox):
    """
    使用刻度计算液位高度。
    args:
        coordinates: 刻度的坐标点，格式如 {"0": [300, 600],....,"1": [600, 200]} 至少包含最小、最大两个刻度
        oil_bbox: 检测出的液位框，格式为[x0, y0, x1, y1]
    """

    if len(coordinates) < 2:
        return None

    for k, v in coordinates.items():     # 取出y值
        coordinates[k] = v[1]           # {'0': 600, '10': 200, '20': 50} 刻度值越大y坐标越小

    # 找出coordinates中距离液位最近的前后两个值
    oil_h = min(oil_bbox[1], oil_bbox[3])    # 返回ymin 对应液位最高点
    y_list = list(coordinates.values())
    # 将配置的刻度分成上、下两个list
    up_list = []
    down_list = []
    for j in y_list:
        if j >= oil_h:
            down_list.append(j)
        else:
            up_list.append(j)
    # 如果某个list为空 那么下方list取y坐标的最大值（刻度的最小值），上list取y坐标的最小值（刻度的最大值）
    if len(down_list) == 0:
        down_list = [max(y_list)]
    elif len(up_list) == 0:
        up_list = [min(y_list)]

    # 返回对应的配置刻度
    y_min = max(up_list)
    y_max = min(down_list) 

    index_down = list(coordinates.values()).index(y_max)
    index_up = list(coordinates.values()).index(y_min)

    _down = list(coordinates.keys())[index_down] # 接近检测值的下刻度
    _up = list(coordinates.keys())[index_up]     # 接近检测值的上刻度
    
    bias = 0.001  # 引入小偏移量
    y_min = y_min - bias  # 对应 上刻度的高度 ymin
    y_max = y_max + bias  # 对应 下刻度的高度 ymax

    # 返回s对应的刻度值，默认竖向
    rate_ = (y_max - oil_h)/(y_max - y_min)
    value = float(_down) + rate_ * (float(_up) - float(_down))  # 刻度下线 + 刻度(上-下)* 百分比 # 刻度下线 + 刻度(上-下)* 百分比

    return value


def inspection_level_gauge(input_data):
    # 提取输入请求信息
    DATA = GetInputData(input_data)
    img_tag = DATA.img_tag
    img_ref = DATA.img_ref
    roi = DATA.roi
    osd = DATA.osd
    pointers = DATA.pointers  # {"0": [1500, 2100],"1":[1680, 780]}
    dp = DATA.dp
    checkpoint = DATA.checkpoint
    an_type = DATA.type
    status_map = DATA.status_map

    # 初始化输出结果
    out_data = {"code": 0, "data": {}, "img_result": DATA.img_tag, "msg": ""}

    # 液位计只有一个roi
    if len(roi) > 1:
        roi_ = {}
        for roi_name in roi:
            roi_[roi_name] = roi[roi_name]
            continue
        print(f'Warning: The number of roi, which must be lower than 1, is equal to {len(roi)}.', end=' ')
        print(f'The input of roi is {roi}, we choose {roi_} for yeweiji now.')
        roi = deepcopy(roi_)
        del roi_

    # 画上点位名称和osd区域
    img_tag_ = img_tag.copy()
    _size = 30
    img_tag_ = img_chinese(img_tag_, checkpoint + an_type, (10, 10), color=(255, 0, 0), size=_size)

    # 如果配置了感兴趣区域，则画出osd区域
    for o_ in osd:
        cv2.rectangle(img_tag_, (int(o_[0]), int(o_[1])),(int(o_[2]), int(o_[3])), (255, 0, 255), thickness=1)
        cv2.putText(img_tag_, "osd", (int(o_[0]), int(o_[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)

    roi_tag_dict, M = roi_registration(img_ref, img_tag, roi)
    if M is None:
        out_data["msg"] = out_data["msg"] + "; Not enough matches are found"
    assert isinstance(roi_tag_dict, dict), f'The type of roi_tag must be dict, but not {type(roi_tag_dict)}.'
    # 画出roi_tag
    roi_name = list(roi_tag_dict.keys())[0]
    roi_tag = roi_tag_dict[roi_name]
    cv2.rectangle(img_tag_, (int(roi_tag[0]), int(roi_tag[1])), (int(roi_tag[2]), int(roi_tag[3])),
                  (255, 0, 255), thickness=1)
    cv2.putText(img_tag_, "roi", (int(roi_tag[0]), int(roi_tag[1])-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)

    # 使用映射变换矫正目标图，并且转换坐标点。
    pointers_tag = conv_coor(pointers, M)
    # for scale in pointers_tag:
    #     coor = pointers_tag[scale]
    #     cv2.circle(img_tag_, (int(coor[0]), int(coor[1])), 1, (255, 0, 255), 8)
    #     cv2.putText(img_tag_, str(scale), (int(coor[0]), int(coor[1])),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), thickness=1)

    if an_type != "level_gauge":
        out_data["msg"] = out_data["msg"] + "type isn't level_gauge; "
        out_data["code"] = 1
        out_data["img_result"] = creat_img_result(input_data, img_tag_) # 返回结果图
        return out_data
    #数字识别液位计
    elif len(pointers_tag)==1:
        bbox_cfg = inference_yolov8(yolov8_jishukuang, img_tag)
        if len(bbox_cfg) < 2:
            out_data["msg"] = out_data["msg"] + "Can not find enough level scale; "
            out_data["code"] = 1
            img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
            out_data["img_result"] = creat_img_result(input_data, img_tag_)  # 返回结果图
            out_data['data']['value'] = None
            # cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag_cfg.jpg"), img_tag_)
            return out_data

            # 检测出的位置按y坐标排序，做640*640填充，二次识别
        coor_list = [item['coor'] for item in bbox_cfg]
        bboxes_list_sort = sorted(coor_list, key=lambda x: x[-1], reverse=False)
        # print("bboxes_list:",bboxes_list)
        value_list=[]
        for coor in bboxes_list_sort:
            img_empty = img_fill(img_tag, coor[0], coor[1], coor[2], coor[3], 640)
            # 二次识别
            bbox_cfg_result = inference_yolov8(yolov8_jishushibie, img_empty)
            bbox_cfg_result = check_iou(bbox_cfg_result, 0.2)
            # print("bbox_cfg_result:",bbox_cfg_result)
            # 按横坐标排序组合结果
            if len(bbox_cfg_result) < 1:
                continue
            label_list = [[item['label'], item['coor']] for item in bbox_cfg_result]
            label_list = sorted(label_list, key=lambda x: x[1][0], reverse=False)
            label = []
            for item in label_list:
                label.append(str(item[0]))
            label.insert(1, ".")
            label = "".join(label)
            label=float(label)
            value_list.append([coor,label])
            cv2.putText(img_tag_, str(label), (int(coor[0]), int(coor[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        if len(value_list)<2:
            out_data["msg"] = out_data["msg"] + "Can not recognize enough level scale; "
            out_data["code"] = 1
            img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
            out_data["img_result"] = creat_img_result(input_data, img_tag_)  # 返回结果图
            out_data['data']['value'] = None
            # cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag_cfg.jpg"), img_tag_)
            return out_data
        v_max=value_list[0][1]
        v_min=value_list[-1][1]
        coor_max=value_list[0][0][3]
        coor_min=value_list[-1][0][3]
        for p in pointers_tag:
            point_coor=pointers_tag[p]
        value=(v_max-v_min)*(point_coor[1]-coor_max)/(coor_min-coor_max)+v_min
        cv2.putText(img_tag_, str(value), (int(point_coor[0]), int(point_coor[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        cfg_out = {"label": "油位","value":round(value,3)}
        out_data["data"] = cfg_out
        out_data["img_result"] = creat_img_result(input_data, img_tag_)
        out_data["msg"] = out_data["msg"] + "success;"
        return out_data


    else:
        out_data["code"] = 0
        yolov8_model = yolov8_yeweiji
        labels_dict = yolov8_model.names
        labels = [labels_dict[i] for i in labels_dict]    # 油位

        # img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=_size)
        # out_data["img_result"] = img2base64(img_tag_)
        
    # 用yolov8检测油位
    cfgs = inference_yolov8(yolov8_yeweiji, img_tag, resize=640) 
    cfgs = check_iou(cfgs, iou_limit=0.5)
    # print("cfgs:", cfgs)

    # 没有检测到目标
    if len(cfgs) == 0:
        out_data["code"] = 1
        out_data["msg"] = out_data["msg"] + "; Not find object"
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=_size)
        out_data["img_result"] = creat_img_result(input_data, img_tag_) # 返回结果图
        return out_data

    # labels 列表 和 color 列表
    colors = color_list(len(labels))
    color_dict = {}
    name_dict = {}
    for i, label in enumerate(labels):
        color_dict[label] = colors[i]
        if len(status_map) > 0 and label in status_map:
            name_dict[label] = status_map[label]
        elif label in config_object_name.OBJECT_MAP[an_type]:
            name_dict[label] = config_object_name.OBJECT_MAP[an_type][label]
        else:
            name_dict[label] = label

    # 画出油位上部液位线
    for cfg in cfgs:
        c = cfg["coor"]
        label = cfg["label"]
        cv2.line(img_tag_, (int(c[0]), int(c[1])), (int(c[2]), int(c[1])), color_dict[label], thickness=2)


       
    # 判断bbox是否在roi中
    cfg_out = {}
    for cfg in cfgs:
        if is_include(cfg["coor"], roi_tag, srate=0.5):
            cfg_out = {"label": name_dict[cfg["label"]], "bbox": cfg["coor"], "score": float(cfg["score"])}
            break

    if len(cfg_out) == 0:
        out_data["code"] = 1
        out_data["msg"] = out_data["msg"] + "; Result not in roi"
        # 选择置信度最高的框作为结果， 舍弃roi
        value = cal_oil_value(pointers_tag, cfgs[0]["coor"])
        if value is not None:
            # 将置信度最高的作为输出
            cfg_out = cfgs[0]

    else:
        # 选择roi内的框为结果
        value = cal_oil_value(pointers_tag, cfg_out["bbox"])

    if value is not None:
        value = round(value, dp)
        cfg_out["value"] = value
        # 可视化最终计算结果
        cv2.putText(img_tag_, str(value), (int(cfg_out["bbox"][0]), int(cfg_out["bbox"][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        if roi_name.startswith("old_roi"):
            out_data["data"] = cfg_out
        else:
            out_data["data"] = {roi_name: [cfg_out]}
    else:
        out_data["code"] = 1
        out_data["msg"] = out_data["msg"] + "at least two coordinates are required"

    out_data["img_result"] = creat_img_result(input_data, img_tag_) # 返回结果图
        
    return out_data


if __name__ == '__main__':

    from lib_help_base import get_save_head, save_input_data, save_output_data
## 图片方式
    # img_tag_file = "/data/home/zgl/datasets/ywj_partitioned/test/20230106165229_5.jpg"
    # img_tag = img2base64(cv2.imread(img_tag_file))
    # input_data = {"image": img_tag, 
    # "config": {
    #     "img_ref": img_tag, 
    #     # "pointers":{"-30": [0.56, 0.67],"0":[0.56, 0.59],
    #     # "70":[0.53, 0.37],"80":[0.54, 0.34],"90":[0.57, 0.313]},
    #     # "bboxes":{"roi":[0.5,0.7,0.56,0.25]},
    #     "pointers":{"20": [0.54, 0.66],"100":[0.54, 0.565],
    #     "160":[0.54, 0.495]},
    #     # "bboxes":{"roi":[0.47,0.21,0.57,0.75]},
    # },
    #     "type": "level_gauge"}
## JSON方式
    f = open("/data/PatrolAi/yolov8/json/yeweiji_old.json","r", encoding='utf-8')
    input_data = json.load(f)

    out_data = inspection_level_gauge(input_data)
    save_dir, name_head = get_save_head(input_data)
    save_input_data(input_data, save_dir, name_head, draw_img=True)
    print(save_dir)
    save_output_data(out_data, save_dir, name_head)

    print("-----------------------------------------------")
    for s in out_data:
        if s != "img_result":
            print(s,":",out_data[s])
    print("----------------------------------------------")
