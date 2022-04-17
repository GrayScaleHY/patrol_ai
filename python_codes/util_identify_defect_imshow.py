from re import S
from lib_inference_yolov5 import load_yolov5_model, inference_yolov5
from lib_sift_match import detect_diff, _resize_feat, sift_match, correct_offset, sift_create
from lib_inference_mrcnn import load_maskrcnn_model, inference_maskrcnn, contour2segment, intersection_arc
from lib_analysis_meter import segment2angle
from lib_image_ops import img_chinese
import numpy as np

yolov5_posun = load_yolov5_model("/data/inspection/yolov5/posun.pt") # 破损类缺陷
yolov5_rotary_switch = load_yolov5_model("/data/inspection/yolov5/rotary_switch.pt") # 切换把手(旋钮开关)
yolov5_led = load_yolov5_model("/data/inspection/yolov5/led.pt") # led灯
yolov5_pressplate = load_yolov5_model("/data/inspection/yolov5/pressplate.pt") # 压板
yolov5_door = load_yolov5_model("/data/inspection/yolov5/door.pt") # 箱门闭合
yolov5_meter = load_yolov5_model("/data/inspection/yolov5/meter.pt") # 表盘
maskrcnn_pointer = load_maskrcnn_model("/data/inspection/maskrcnn/pointer.pth", num_classes=1, score_thresh=0.3) # 加载指针的maskrcnn模型

def indentify_pointer(img_ref, img_tag):
    """
    判断指针读数是否过大
    """
    bbox_cfg_tag = inference_yolov5(yolov5_meter, img_tag)
    if len(bbox_cfg_tag) == 0:
        return []
    
    bbox_cfg_ref = inference_yolov5(yolov5_meter, img_ref)
    if len(bbox_cfg_ref) == 0:
        return []

    ## 取最大的bbox
    s_max = 0 
    for cfg in bbox_cfg_tag:
        c = cfg["coor"]
        s = (c[2] - c[0]) * (c[3] - c[1])
        if s > s_max:
            s_max = s
            label_tag = cfg["label"]
            coor_tag = c
    
    s_max = 0
    for cfg in bbox_cfg_ref:
        c = cfg["coor"]
        s = (c[2] - c[0]) * (c[3] - c[1])
        if s > s_max:
            s_max = s
            label_ref = cfg["label"]
            coor_ref = c

    img_tag_meter = img_tag[coor_tag[1]:coor_tag[3], coor_tag[0]:coor_tag[2],:]
    img_ref_meter = img_ref[coor_ref[1]:coor_ref[3], coor_ref[0]:coor_ref[2],:]
    contours, boxes, (masks, classes, scores) = inference_maskrcnn(maskrcnn_pointer, img_tag_meter)
    segments_tag = contour2segment(contours, boxes)
    contours, boxes, (masks, classes, scores) = inference_maskrcnn(maskrcnn_pointer, img_ref_meter)
    segments_ref = contour2segment(contours, boxes)
    if len(segments_ref) == 0 or len(segments_tag) == 0:
        return []
    else:
        seg_tag = segments_tag[0]
        seg_ref = segments_ref[0]
    
    xo = (coor_tag[2]-coor_tag[0]) / 2; yo = (coor_tag[3]-coor_tag[1]) / 2
    if (seg_tag[0]-xo)**2+(seg_tag[1]-yo)**2 < (seg_tag[2]-xo)**2+(seg_tag[3]-yo)**2:
        seg_tag = [seg_tag[2], seg_tag[3], seg_tag[0], seg_tag[1]]
    
    xo = (coor_ref[2]-coor_ref[0]) / 2; yo = (coor_ref[3]-coor_ref[1]) / 2
    if (seg_ref[0]-xo)**2+(seg_ref[1]-yo)**2 < (seg_ref[2]-xo)**2+(seg_ref[3]-yo)**2:
        seg_ref = [seg_ref[2], seg_ref[3], seg_ref[0], seg_ref[1]]
    
    angle_tag = segment2angle(seg_tag[:2], seg_tag[-2:])
    angle_ref = segment2angle(seg_ref[:2], seg_ref[-2:])

    angle_dif1 = abs(angle_tag - angle_ref)
    angle_dif2 = 360 - abs(angle_tag - angle_ref)
    angle_dif = min(angle_dif1, angle_dif2)
    if angle_dif < 30:
        return []
    xmin = int(min(seg_tag[0]+coor_tag[0], seg_tag[2]+coor_tag[0]))
    ymin = int(min(seg_tag[1]+coor_tag[1], seg_tag[3]+coor_tag[1]))
    xmax = int(max(seg_tag[0]+coor_tag[0], seg_tag[2]+coor_tag[0]))
    ymax = int(max(seg_tag[1]+coor_tag[1], seg_tag[3]+coor_tag[1]))

    return [xmin, ymin, xmax, ymax]


def identify_yolov5(bbox_cfg_ref, bbox_cfg_tag):
    """
    判断ref的目标物信息和tag的目标物信息是否一致，不一致的话返回不一致的目标框列表。
    """
    if len(bbox_cfg_tag) == 0:
        return []
    
    if len(bbox_cfg_ref) == 0:
        return []
    
    ## 判断对应位置的目标物是否标签一致，如果不一致,将目标看放入到tag_diff中。
    tag_diff = []
    for cfg_tag in bbox_cfg_tag:
        c_tag = cfg_tag["coor"]
        xo = (c_tag[2] + c_tag[0]) / 2; yo = (c_tag[3] + c_tag[1]) / 2
        for cfg_ref in bbox_cfg_ref:
            c_ref = cfg_ref["coor"]
            if c_ref[0] < xo < c_ref[2] and c_ref[1] < yo < c_ref[3]:
                if cfg_ref["label"] != cfg_tag["label"]:
                    tag_diff.append(c_tag)
    
    if len(tag_diff) == 0:
        return []
    else:
        return tag_diff


def identify_defect(img_ref, feat_ref, img_tag, feat_tag):
    """
    判别算法
    """
    tag_diff = []
    img_tag_ = img_tag.copy()
    img_ref_ = img_ref.copy()
    
    ## 基于tag对ref进行矫正
    M = sift_match(feat_tag, feat_ref, ratio=0.5, ops="Affine")
    img_ref = correct_offset(img_ref, M)

    # ## 箱门闭合异常判别
    bbox_cfg_tag = inference_yolov5(yolov5_door, img_tag)
    if len(bbox_cfg_tag) != 0:
        bbox_cfg_ref = inference_yolov5(yolov5_door, img_ref)
        tag_diff = identify_yolov5(bbox_cfg_ref, bbox_cfg_tag)
        if len(tag_diff) != 0:

            for d_ in tag_diff:
                cv2.rectangle(img_tag_, (d_[0], d_[1]),(d_[2], d_[3]), (0,0,255), thickness=2)
                img_tag_ = img_chinese(img_tag_, "箱门类异常", (d_[0], d_[1]+10), color=(0,0,255), size=20)
            # return tag_diff

    # ## 压板异常判别
    bbox_cfg_tag = inference_yolov5(yolov5_pressplate, img_tag)
    if len(bbox_cfg_tag) != 0:
        bbox_cfg_ref = inference_yolov5(yolov5_pressplate, img_ref)
        tag_diff = identify_yolov5(bbox_cfg_ref, bbox_cfg_tag)
        if len(tag_diff) != 0:

            for d_ in tag_diff:
                cv2.rectangle(img_tag_, (d_[0], d_[1]),(d_[2], d_[3]), (0,255,0), thickness=2)
                img_tag_ = img_chinese(img_tag_, "压板类异常", (d_[0], d_[1]+20), color=(0,255,0), size=20)
            # return tag_diff

    ## 指示灯异常判别
    bbox_cfg_tag = inference_yolov5(yolov5_led, img_tag)
    if len(bbox_cfg_tag) != 0:
        bbox_cfg_ref = inference_yolov5(yolov5_led, img_ref)
        tag_diff = identify_yolov5(bbox_cfg_ref, bbox_cfg_tag)
        if len(tag_diff) != 0:

            for d_ in tag_diff:
                cv2.rectangle(img_tag_, (d_[0], d_[1]),(d_[2], d_[3]), (255,0,0), thickness=2)
                img_tag_ = img_chinese(img_tag_, "指示灯类异常", (d_[0], d_[1]+20), color=(255,0,0), size=20)
            # return tag_diff

    ## 破损类异常判别
    bbox_cfg_tag = inference_yolov5(yolov5_posun, img_tag)
    if len(bbox_cfg_tag) != 0:
        tag_diff = []
        for cfg in bbox_cfg_tag:
            tag_diff.append(cfg["coor"])

        for d_ in tag_diff:
            cv2.rectangle(img_tag_, (d_[0], d_[1]),(d_[2], d_[3]), (125,125,125), thickness=2)
            img_tag_ = img_chinese(img_tag_, "破损类异常", (d_[0], d_[1]+20), color=(125,125,125), size=20)
        # return tag_diff

    ## 旋钮开关异常判别
    bbox_cfg_tag = inference_yolov5(yolov5_rotary_switch, img_tag)
    if len(bbox_cfg_tag) != 0:
        bbox_cfg_ref = inference_yolov5(yolov5_rotary_switch, img_ref)
        tag_diff = identify_yolov5(bbox_cfg_ref, bbox_cfg_tag)
        if len(tag_diff) != 0:

            for d_ in tag_diff:
                cv2.rectangle(img_tag_, (d_[0], d_[1]),(d_[2], d_[3]), (255,255,0), thickness=2)
                img_tag_ = img_chinese(img_tag_, "旋钮开关类异常", (d_[0], d_[1]+20), color=(255,255,0), size=20)
            # return tag_diff

    ## 指针读数变化太大
    rec = indentify_pointer(img_ref, img_tag)
    if len(rec) != 0:  
        tag_diff = [rec]
        d_ = rec
 
        cv2.rectangle(img_tag_, (d_[0], d_[1]),(d_[2], d_[3]), (255,0,255), thickness=2)
        img_tag_ = img_chinese(img_tag_, "指针变化类异常", (d_[0], d_[1]+20), color=(0,255,255), size=20)
        # return tag_diff

    # ## 像素相减类异常
    rec = detect_diff(img_ref_, feat_ref, img_tag, feat_tag)
    if len(rec) != 0:  
        tag_diff = [rec]

        d_ = rec
        cv2.rectangle(img_tag_, (d_[0], d_[1]),(d_[2], d_[3]), (255,0,255), thickness=2)
        img_tag_ = img_chinese(img_tag_, "像素相减类异常", (d_[0], d_[1]+20), color=(255,0,255), size=20)
        # return tag_diff
    
    return img_tag_
    # return tag_diff

if __name__ == '__main__':
    import glob
    import os
    import cv2
    import time
    import shutil

    in_dir = "test/panbie"  # 判别图像存放目录
    out_dir = "test/panbie_result" # 输出结果目录
    resize_rate = 0.5

    os.makedirs(out_dir, exist_ok=True)
    for ref_file in glob.glob(os.path.join(in_dir, "*_normal.jpg")):

        shutil.copy(ref_file, os.path.join(out_dir, os.path.basename(ref_file)))

        file_id = os.path.basename(ref_file).split("_")[0]
        img_ref = cv2.imread(ref_file) 

        H, W = img_ref.shape[:2]  ## resize
        img_ref = cv2.resize(img_ref, (int(W * resize_rate), int(H * resize_rate)))

        feat_ref = sift_create(img_ref)

        for tag_file in glob.glob(os.path.join(in_dir, file_id + "_*.jpg")):
            tag_name = os.path.basename(tag_file)

            if tag_file.endswith("normal.jpg"):
                continue
            img_tag = cv2.imread(tag_file)

            H, W = img_tag.shape[:2]  ## resize
            img_tag = cv2.resize(img_tag, (int(W * resize_rate), int(H * resize_rate)))

            feat_tag = sift_create(img_tag)

            img_tag_ = identify_defect(img_ref, feat_ref, img_tag, feat_tag)

            out_file = os.path.join(out_dir, tag_name)  # 结果显示
            cv2.imwrite(out_file,img_tag_)

