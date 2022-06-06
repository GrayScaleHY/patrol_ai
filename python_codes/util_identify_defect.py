"""
用于判别算法的测试。
"""

from lib_inference_yolov5 import inference_yolov5
from lib_sift_match import detect_diff, sift_match, correct_offset, sift_create
from lib_inference_mrcnn import inference_maskrcnn, contour2segment
from lib_analysis_meter import segment2angle
from lib_image_ops import img_chinese
import glob
import os
import cv2
import time
import shutil
import numpy as np

## 二次设备， coco， 17类缺陷， 表计， 指针
from config_load_models_var import yolov5_ErCiSheBei, yolov5_coco, yolov5_rec_defect, yolov5_meter, maskrcnn_pointer

def indentify_pointer(img_ref, img_tag):
    """
    判断指针读数是否过大
    """
    cfgs_tag = inference_yolov5(yolov5_meter, img_tag)
    if len(cfgs_tag) == 0:
        return []
    
    cfgs_ref = inference_yolov5(yolov5_meter, img_ref)
    if len(cfgs_ref) == 0:
        return []

    cfg_tag = cfgs_tag[0]; cfg_ref = cfgs_ref[0]
    c_tag = cfg_tag["coor"]; c_ref = cfg_ref["coor"]
    tag_meter = img_tag[c_tag[1]:c_tag[3], c_tag[0]:c_tag[2],:]
    ref_meter = img_ref[c_ref[1]:c_ref[3], c_ref[0]:c_ref[2],:]

    ## 求出指针
    contours, boxes, (masks, classes, scores) = inference_maskrcnn(maskrcnn_pointer, tag_meter)
    segs_tag = contour2segment(contours, boxes)
    if len(segs_tag) == 0:
        return []
    contours, boxes, (masks, classes, scores) = inference_maskrcnn(maskrcnn_pointer, ref_meter)
    segs_ref = contour2segment(contours, boxes)
    if len(segs_ref) == 0:
        return []
    # seg_tag = segs_tag[0]
    # seg_ref = segs_ref[0]
    
    ## 计算图片中的指针角度
    angles_tag = []
    xo = tag_meter.shape[1] / 2; yo = tag_meter.shape[0] / 2
    for seg_tag in segs_tag:
        if (seg_tag[0]-xo)**2+(seg_tag[1]-yo)**2 < (seg_tag[2]-xo)**2+(seg_tag[3]-yo)**2:
            seg_tag = [seg_tag[2], seg_tag[3], seg_tag[0], seg_tag[1]]
        angle = segment2angle(seg_tag[:2], seg_tag[-2:])
        angles_tag.append(angle)
    
    angles_ref = []
    xo = ref_meter.shape[1] / 2; yo = ref_meter.shape[0] / 2
    for seg_ref in segs_ref:
        if (seg_ref[0]-xo)**2+(seg_ref[1]-yo)**2 < (seg_ref[2]-xo)**2+(seg_ref[3]-yo)**2:
            seg_ref = [seg_ref[2], seg_ref[3], seg_ref[0], seg_ref[1]]
        angle = segment2angle(seg_ref[:2], seg_ref[-2:])
        angles_ref.append(angle)
    
    ## 判断angles_tag中是否存在指针相对于angles_ref角度偏差大于15的
    ang_dif = True
    for angle_tag in angles_tag:
        ang_sam = False
        for angle_ref in angles_ref:
            angle_dif1 = abs(angle_tag - angle_ref)
            angle_dif2 = 360 - abs(angle_tag - angle_ref)
            angle_dif = min(angle_dif1, angle_dif2)
            if angle_dif < 15:
                ang_sam = True
        if not ang_sam:
            ang_dif = False
    if ang_dif:
        return []

    # seg_all = []
    # for seg in segs_tag:
    #     seg_all.append([seg[0]+c_tag[0], seg[1]+c_tag[1]])
    #     seg_all.append([seg[2]+c_tag[0], seg[3]+c_tag[1]])
    # for seg in segs_ref:
    #     seg_all.append([seg[0]+c_ref[0], seg[1]+c_ref[1]])
    #     seg_all.append([seg[2]+c_ref[0], seg[3]+c_ref[1]])
    # ca = np.array(seg_all, dtype=int)
    # xmin = np.min(ca[:,0])
    # ymin = np.min(ca[:,1])
    # xmax = np.max(ca[:,0])
    # ymax = np.max(ca[:,1])

    return c_tag

def labels_diff_area(cfgs_ref, cfgs_tag):
    """
    判断cfgs_ref和cfgs_tag是否发生了位置变化。
    args:
        bbox_cfg_ref: 基准图的yolov5推理信息，格式为[{"label": "", "coor": [x0, y0, x1, y1], "score": float}, {}, ..]
        bbox_cfg_tag: 待分析图的yolov5推理信息，格式为[{"label": "", "coor": [x0, y0, x1, y1], "score": float}, {}, ..]
    return:
        tag_diff: 不一致目标框,[xmin, ymin, xmax, ymax]
    """
    
    ## 判断对应位置的目标物是否标签一致，如果不一致,将目标看放入到tag_diff中。
    diff_boxes = []

    ## 判断tag中的目标在ref中是否有改变
    for cfg in cfgs_tag:
        c = cfg["coor"]; l = cfg["label"]
        xo = (c[2] + c[0]) / 2; yo = (c[3] + c[1]) / 2
        is_exist = False
        for cfg_ in cfgs_ref:
            c_ = cfg_["coor"]; l_ = cfg_["label"]
            if l == l_ and  c_[0] < xo < c_[2] and c_[1] < yo < c_[3]:
                is_exist = True
        if not is_exist:
            diff_boxes.append(c)
    
    ## 判断ref中的目标在tag中是否有改变
    for cfg in cfgs_ref:
        c = cfg["coor"]; l = cfg["label"]
        xo = (c[2] + c[0]) / 2; yo = (c[3] + c[1]) / 2
        is_exist = False
        for cfg_ in cfgs_tag:
            c_ = cfg_["coor"]; l_ = cfg_["label"]
            if l == l_ and  c_[0] < xo < c_[2] and c_[1] < yo < c_[3]:
                is_exist = True
        if not is_exist:
            diff_boxes.append(c)
    
    if len(diff_boxes) == 0:
        return []
    
    ## 将多个box合并
    d = np.array(diff_boxes, dtype=int)
    diff_area = [np.min(d[:,0]), np.min(d[:,1]), np.max(d[:,2]), np.max(d[:,3])]
    return diff_area

def identify_defect(img_ref, feat_ref, img_tag, feat_tag):
    """
    判别算法
    args:
        img_ref: 基准图
        feat_ref: 基准图的sift特征
        img_tag: 待分析图
        feat_tag: 待分析图的特征
    return:
        diff_area:不一致目标框,格式为[xmin, ymin, xmax, ymax]
    """
    tag_diff = []

    ## 将图片中osd区域中的sift特征点去掉。
    # H, W = img_ref.shape[:2]
    # osd_boxes = [[0, 0, 1, 0.12], [0, 0.88, 1, 1]] # 将图像上下12%的区域内sift特征点去掉
    # # osd_boxes = [] # 不处理osd区域
    # rm_regs = []
    # for b in osd_boxes:
    #     b_ = [int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)]
    #     rm_regs.append(b_)
    
    ## 基于tag对ref进行矫正
    M = sift_match(feat_tag, feat_ref, ratio=0.5, ops="Affine")
    img_ref, cut = correct_offset(img_ref, M, b=True)

    # ## 用yolov5检测待测图和基准图的目标物和状态
    # 缺陷
    pre_labels = ["yw_gkxfw", "yw_nc", "bj_bpps", "jyz_pl", "hxq_gjtps"] 
    cfgs_tag = inference_yolov5(yolov5_rec_defect, img_tag, resize=1280, conf_thres=0.7, iou_thres=0.2, pre_labels=pre_labels)
    cfgs_ref = inference_yolov5(yolov5_rec_defect, img_ref, resize=1280, conf_thres=0.7, iou_thres=0.2, pre_labels=pre_labels)
    diff_area = labels_diff_area(cfgs_ref, cfgs_tag)
    if len(diff_area) != 0:
        return diff_area
        
    # coco
    pre_labels = ["person"]
    cfgs_tag = inference_yolov5(yolov5_coco, img_tag, resize=640, conf_thres=0.85, iou_thres=0.2, pre_labels=pre_labels)
    cfgs_ref = inference_yolov5(yolov5_coco, img_ref, resize=640, conf_thres=0.85, iou_thres=0.2, pre_labels=pre_labels)
    diff_area = labels_diff_area(cfgs_ref, cfgs_tag)
    if len(diff_area) != 0:
        return diff_area

    # 二次设备
    pre_labels = ["kgg_ybh", "kgg_ybf", "kqkg_hz", "kqkg_fz", "xnkg_s", "xnkg_zs", "xnkg_ys", "xnkg_z", "zsd_l", "zsd_m"] 
    cfgs_tag = inference_yolov5(yolov5_ErCiSheBei, img_tag, resize=640, conf_thres=0.75, iou_thres=0.2, pre_labels=pre_labels)
    cfgs_ref = inference_yolov5(yolov5_ErCiSheBei, img_ref, resize=640, conf_thres=0.75, iou_thres=0.2, pre_labels=pre_labels)
    diff_area = labels_diff_area(cfgs_ref, cfgs_tag)
    if len(diff_area) != 0:
        return diff_area

    
    ## 计算指针是否读数变化幅度过大
    diff_area = indentify_pointer(img_ref, img_tag)
    if len(diff_area) != 0:  
        return diff_area

    ## 像素相减类异常
    img_tag = img_tag[cut[1]:cut[3], cut[0]:cut[2], :]
    img_ref = img_ref[cut[1]:cut[3], cut[0]:cut[2], :]
    diff_area = detect_diff(img_ref, img_tag)
    if len(diff_area) != 0:  
        diff_area = [diff_area[0] + cut[0], diff_area[1] + cut[1], diff_area[2] + cut[0], diff_area[3] + cut[1]]
        return diff_area
    
    return tag_diff

if __name__ == '__main__':
    # from lib_sift_match import init_data
    # init_data()

    in_dir = "test/panbie"  # 判别测试图片存放目录
    out_dir = "test/panbie_result" # 判别算法输出目录
    resize_max = 1280   ## 图像最长边缩放到多少

    start_all = time.time()

    os.makedirs(out_dir, exist_ok=True)
    for ref_file in glob.glob(os.path.join(in_dir, "*_normal.jpg")):

        file_id = os.path.basename(ref_file).split("_")[0]
        img_ref = cv2.imread(ref_file) 

        # # resize, 降低分别率，加快特征提取的速度。
        # H, W = img_ref.shape[:2]
        # resize_rate = max(H, W) / resize_max  ## 缩放倍数
        # img_ref = cv2.resize(img_ref, (int(W / resize_rate), int(H / resize_rate)))

        feat_ref = sift_create(img_ref) # 提取sift特征

        for tag_file in glob.glob(os.path.join(in_dir, file_id + "_*.jpg")):
            loop_start = time.time()
            print("-------------------------------")
            
            tag_name = os.path.basename(tag_file)

            if tag_file.endswith("normal.jpg"):
                continue
            print(tag_file)
            img_tag = cv2.imread(tag_file)

            # H, W = img_tag.shape[:2]  ## resize
            # img_tag = cv2.resize(img_tag, (int(W / resize_rate), int(H / resize_rate)))
            feat_tag = sift_create(img_tag) # 提取sift特征
            tag_diff = identify_defect(img_ref, feat_ref, img_tag, feat_tag) # 判别算法

            ## 将tag_diff还原回原始大小
            # tag_diff = [int(d * resize_rate) for d in tag_diff]
            print(tag_diff)

            ## 将结果写成txt
            if len(tag_diff) == 0:
                # s = "1," + tag_name + ",0,0\n"
                s = "1," + tag_name + ",0,0,0,0,0\n"
            else:
                tag_diff = [str(tag_diff[0]), str(tag_diff[1]), str(tag_diff[2]), str(tag_diff[3])]
                s = "1," + tag_name + ",1," + ",".join(tag_diff) + "\n"
            out_file = os.path.join(out_dir, tag_name[:-4] + ".txt")
            f = open(out_file, "w", encoding='utf-8')
            f.write("ID,PATH,TYPE,XMIN,YMIN,XMAX,YMAX\n")
            f.write(s)
            f.close()
            print("loop time:", time.time() - loop_start)
            print("--------------------------")

    print("Total spend times:", time.time() - start_all)