import os
import time
import cv2
import json
from util_identify_defect import indentify_door, indentify_led, indentify_pointer, indentify_pressplate
from lib_image_ops import base642img, img2base64, img_chinese
from lib_help_base import color_list
from lib_sift_match import _resize_feat, detect_diff
import numpy as np

def indentify_defect(img_ref, img_tag):
    
    ## 箱门闭合异常
    rec = indentify_door(img_ref, img_tag)
    if len(rec) != 0:
        return rec
    
    ## 仪表读数变化太大
    rec = indentify_pointer(img_ref, img_tag)
    if len(rec) != 0:
        return rec
    
    ## 压板
    rec = indentify_pressplate(img_ref, img_tag)
    if len(rec) != 0:
        return rec

    ## led灯颜色变化
    rec = indentify_led(img_ref, img_tag)
    if len(rec) != 0:
        return rec
    
    ref, rate, feat_ref = _resize_feat(img_ref)
    tag, rate, feat_tag = _resize_feat(img_tag)
    rec = detect_diff(img_ref, ref, feat_ref, img_tag, tag, feat_tag, rate)

    return rec

def get_input_data(input_data):
    """
    提取input_data中的信息。
    return:
        img_tag: 目标图片数据
        img_ref: 模板图片数据
    """

    img_tag = base642img(input_data["image"])

    ## 是否有模板图
    img_ref = None
    if "img_ref" in input_data["config"]:
        if isinstance(input_data["config"]["img_ref"], str):
            img_ref = base642img(input_data["config"]["img_ref"]) 

    return img_tag, img_ref


def inspection_object_detection(input_data):
    """
    yolov5的目标检测推理。
    """
    ## 将输入请求信息可视化
    TIME_START = time.strftime("%m-%d-%H-%M-%S") 
    save_path = os.path.join("inspection_result", input_data["type"], TIME_START)
    os.makedirs(save_path, exist_ok=True)
    f = open(os.path.join(save_path, "input_data.json"), "w")
    json.dump(input_data, f, ensure_ascii=False)  # 保存输入信息json文件
    f.close()

    ## 初始化输入输出信息。
    img_tag, img_ref = get_input_data(input_data)
    out_data = {"code": 0, "data":[], "img_result": input_data["image"], "msg": "Success request object detect; "} # 初始化out_data
    if img_ref is None:
        out_data["msg"] = out_data["msg"] + "; img_ref not exist;"
        return out_data

    img_tag_ = img_tag.copy()
    cv2.imwrite(os.path.join(save_path, "img_tag.jpg"), img_tag) # 将输入图片可视化
    cv2.imwrite(os.path.join(save_path, "img_ref.jpg"), img_ref) # 将输入图片可视化

    rec = indentify_defect(img_ref, img_tag)

    if len(rec) == 0:
        label = "0"
        img_tag_ = img_chinese(img_tag_, "正常", (10, 10), (0, 255, 0), size=10)
    else:
        label = "1"
        cv2.rectangle(img_tag_, (int(rec[0]), int(rec[1])),(int(rec[2]), int(rec[3])), "异常", thickness=2)
        img_tag_ = img_chinese(img_tag_, "异常", (int(rec[0])+10, int(rec[1])+10), (0,0,255), size=10)
    
    out_data["data"] = [{"label": label, "bbox": rec}]

    ## 可视化计算结果
    f = open(os.path.join(save_path, "out_data.json"), "w")
    json.dump(out_data, f, ensure_ascii=False, indent=2)  # 保存输入信息json文件
    f.close()
    cv2.imwrite(os.path.join(save_path, "img_tag_cfg.jpg"), img_tag_)

    ## 输出可视化结果的图片。
    out_data["img_result"] = img2base64(img_tag_)

    return out_data

if __name__ == '__main__':
    a = 1