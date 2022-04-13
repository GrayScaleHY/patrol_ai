from re import S
from lib_inference_yolov5 import load_yolov5_model, inference_yolov5
from lib_sift_match import detect_diff, _resize_feat
from lib_inference_mrcnn import load_maskrcnn_model, inference_maskrcnn, contour2segment, intersection_arc
from lib_analysis_meter import segment2angle
import numpy as np

yolov5_led = load_yolov5_model("/data/inspection/yolov5/led.pt") # led灯
yolov5_pressplate = load_yolov5_model("/data/inspection/yolov5/pressplate.pt") # 压板
yolov5_door = load_yolov5_model("/data/inspection/yolov5/door.pt") # 箱门闭合
yolov5_meter = load_yolov5_model("/data/inspection/yolov5/meter.pt") # 表盘
maskrcnn_pointer = load_maskrcnn_model("/data/inspection/maskrcnn/pointer.pth", num_classes=1, score_thresh=0.3) # 加载指针的maskrcnn模型

def indentify_door(img_ref, img_tag):
    """
    判断是否为箱门闭合异常
    """
    bbox_cfg_tag = inference_yolov5(yolov5_door, img_tag)
    if len(bbox_cfg_tag) == 0:
        return []
    
    bbox_cfg_ref = inference_yolov5(yolov5_door, img_ref)
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
    
    if label_tag == label_ref:
        return []
    else:
        return coor_tag


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


def indentify_pressplate(img_ref, img_tag):
    bbox_cfg_tag = inference_yolov5(yolov5_pressplate, img_tag)
    if len(bbox_cfg_tag) == 0:
        return []
    
    bbox_cfg_ref = inference_yolov5(yolov5_pressplate, img_ref)
    if len(bbox_cfg_ref) == 0:
        return []
    
    if len(bbox_cfg_ref) != len(bbox_cfg_tag):
        return []
    
    ## 判断对应位置的压板是否有异样， 适用于只有一个压板变化了
    tag_diff = []
    for cfg_tag in bbox_cfg_tag:
        c_tag = cfg_tag["coor"]
        xo = (c_tag[2] - c_tag[0]) / 2; yo = (c_tag[3] - c_tag[1]) / 2
        for cfg_ref in bbox_cfg_ref:
            c_ref = cfg_ref["coor"]
            if c_ref[0] < xo < c_ref[2] and c_ref[1] < yo < c_ref[3]:
                if cfg_ref["label"] != cfg_tag["label"]:
                    tag_diff.append(c_tag)
    
    if len(tag_diff) != 1:
        return []
    else:
        return tag_diff[0]

    ## 适用于会同时出现多个压板状态变化的情况
    # bbox_tag_on = []
    # bbox_tag_off = []
    # for cfg in bbox_cfg_tag:
    #     if cfg["label"] == "pressplate_on":
    #         bbox_tag_on.append(cfg["coor"])
    #     else:
    #         bbox_tag_off.append(cfg["coor"])
    # bbox_ref_on = []
    # bbox_ref_off = []
    # for cfg in bbox_cfg_ref:
    #     if cfg["label"] == "pressplate_on":
    #         bbox_ref_on.append(cfg["coor"])
    #     else:
    #         bbox_ref_off.append(cfg["coor"])
    
    # if len(bbox_tag_on) > len(bbox_ref_on) and len(bbox_tag_off) < len(bbox_ref_off):
    #     bbox_tag = bbox_tag_on
    #     bbox_ref = bbox_ref_on
    # elif len(bbox_tag_on) < len(bbox_ref_on) and len(bbox_tag_off) > len(bbox_ref_off):
    #     bbox_tag = bbox_tag_off
    #     bbox_ref = bbox_ref_off
    # else:
    #     return []
    
    # tag_diff = []
    # if len(bbox_ref) == 0:
    #     tag_diff = bbox_tag
    # else:
    #     for tag in bbox_tag:
    #         xo = (tag[2] - tag[0]) / 2; yo = (tag[3] - tag[0]) / 2
    #         for ref in bbox_ref:
    #             if ref[0] < xo < ref[2] and ref[1] < yo < ref[3]:
    #                 continue
    #             if tag not in tag_diff:
    #                 tag_diff.append(tag)
    # if len(tag_diff) == 0:
    #     return []
    # else:
    #     tag_diff = np.array(tag_diff, dtype=int)
    #     xmin = np.min(tag_diff[:,0])
    #     ymin = np.min(tag_diff[:,1])
    #     xmax = np.max(tag_diff[:,0])
    #     ymax = np.max(tag_diff[:,1])
    #     return [xmin, ymin, xmax, ymax]


def indentify_led(img_ref, img_tag):
    """
    判别led灯是否异常
    """
    bbox_cfg_tag = inference_yolov5(yolov5_led, img_tag)
    if len(bbox_cfg_tag) == 0:
        return []
    
    bbox_cfg_ref = inference_yolov5(yolov5_led, img_ref)
    if len(bbox_cfg_ref) == 0:
        return []
    
    if len(bbox_cfg_ref) != len(bbox_cfg_tag):
        return []
    
    ## 判断对应位置的led灯是否有异样， 适用于只有一个灯变化了
    tag_diff = []
    for cfg_tag in bbox_cfg_tag:
        c_tag = cfg_tag["coor"]
        xo = (c_tag[2] - c_tag[0]) / 2; yo = (c_tag[3] - c_tag[1]) / 2
        for cfg_ref in bbox_cfg_ref:
            c_ref = cfg_ref["coor"]
            if c_ref[0] < xo < c_ref[2] and c_ref[1] < yo < c_ref[3]:
                if cfg_ref["label"] != cfg_tag["label"]:
                    tag_diff.append(c_tag)
    
    if len(tag_diff) != 1:
        return []
    else:
        return tag_diff[0]


def indentify_defect(img_ref, ref, feat_ref, img_tag, tag, feat_tag, rate):
    
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
    
    rec = detect_diff(img_ref, ref, feat_ref, img_tag, tag, feat_tag, rate)

    return rec

if __name__ == '__main__':
    import glob
    import os
    import cv2


    in_dir = "pb_nj"
    out_dir = "tuilishuju_output"
    os.makedirs(out_dir, exist_ok=True)
    for ref_file in glob.glob(os.path.join(in_dir, "*_normal.jpg")):

        file_id = os.path.basename(ref_file).split("_")[0]
        img_ref = cv2.imread(ref_file)
        ref, rate, feat_ref = _resize_feat(img_ref)

        for tag_file in glob.glob(os.path.join(in_dir, file_id + "_*.jpg")):
            tag_name = os.path.basename(tag_file)
            if tag_file.endswith("normal.jpg"):
                continue
            
            out_file = os.path.join(out_dir, tag_name[:-4] + ".jpg")
            f = open(out_file, "w", encoding='utf-8')
            f.write("ID,PATH,TYPE,XMIN,YMIN,XMAX,YMAX\n")

            img_tag = cv2.imread(tag_file)
            tag, rate, feat_tag = _resize_feat(img_tag)
            rec = indentify_defect(img_ref, ref, feat_ref, img_tag, tag, feat_tag, rate)

            if len(rec) == 0:
                s = "1," + tag_name +"1"
            
            



