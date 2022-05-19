from lib_inference_yolov5 import load_yolov5_model, inference_yolov5
from lib_sift_match import detect_diff, sift_match, correct_offset, sift_create
from lib_inference_mrcnn import load_maskrcnn_model, inference_maskrcnn, contour2segment
from lib_analysis_meter import segment2angle
from lib_image_ops import img_chinese
import glob
import os
import cv2
import time
import shutil
import numpy as np

yolov5_ErCiSheBei = load_yolov5_model("/data/inspection/yolov5/ErCiSheBei.pt") ## 二次设备状态模型
yolov5_coco = load_yolov5_model("/data/inspection/yolov5/coco.pt") # coco模型
labels = yolov5_coco.module.names if hasattr(yolov5_coco, 'module') else yolov5_coco.names
print(labels)
yolov5_rec_defect = load_yolov5_model("/data/inspection/yolov5/rec_defect_x6.pt") # x6的17类缺陷模型
yolov5_meter = load_yolov5_model("/data/inspection/yolov5/meter.pt") # 表盘
maskrcnn_pointer = load_maskrcnn_model("/data/inspection/maskrcnn/pointer.pth", num_classes=1, score_thresh=0.3) # 加载指针的maskrcnn模型

in_dir = "test/panbie"
out_dir = "test/panbie_result"

for img_file in glob.glob(os.path.join(in_dir, "*.jpg")):
    print(img_file)

    img = cv2.imread(img_file)

    cfg_all = []

    cfgs = inference_yolov5(yolov5_ErCiSheBei, img)
    for cfg in cfgs:
        l = cfg["label"]
        if l != "xmbhyc" and l != "ys" and l != "fpq_h" and l != "fpq_f" and l != "fpq_jd": 
            cfg_all.append(cfg)
    
    cfgs = inference_yolov5(yolov5_meter, img)
    for cfg in cfgs:
        cfg_all.append(cfg)
        c = cfg["coor"]
        img_tag_meter = img[c[1]:c[3], c[0]:c[2], :]
        contours, boxes, (masks, classes, scores) = inference_maskrcnn(maskrcnn_pointer, img_tag_meter)
        for box in boxes:
            box = [box[0] + c[0], box[1] + c[1], box[2] + c[0], box[3] + c[1]]
            cfg_all.append({"label": "pointer", "coor": box})

    cfgs = inference_yolov5(yolov5_rec_defect, img, resize=1280)
    for cfg in cfgs:
        l = cfg["label"]
        if l == "yw_gkxfw" or l == "yw_nc" or l == "jyz_pl" or l == "bj_bpps" or l == "hxq_gjtps" or l == "xmbhyc":
            cfg_all.append(cfg)
    
    cfgs = inference_yolov5(yolov5_coco, img)
    for cfg in cfgs:
        l = cfg["label"]
        if l == "person" or l == "car" or l == "bus" or l == "truck":
            cfg_all.append(cfg)
        
    
    for cfg in cfg_all:
        l = cfg["label"]
        c = cfg["coor"]
        cv2.rectangle(img, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), (0, 0, 255), thickness=2)
        cv2.putText(img, l, (int(c[0]), int(c[1])+20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
    print(cfg_all)
    cv2.imwrite(os.path.join(out_dir, os.path.basename(img_file)), img)
    


    