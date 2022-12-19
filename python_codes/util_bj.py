import cv2
import time
import os
import json
from app_pointer import pointer_detect, select_pointer, conv_coor, cal_base_scale, add_head_end_ps, is_include
from lib_sift_match import sift_match, convert_coor, sift_create
from lib_image_ops import base642img, img2base64
import numpy as np
from lib_image_ops import img_chinese
import glob
from config_load_models_var import yolov5_ShuZiBiaoJi
from lib_inference_yolov5 import load_yolov5_model, inference_yolov5
from lib_inference_mrcnn import load_maskrcnn_model, inference_maskrcnn
import argparse

maskrcnn_oil = load_maskrcnn_model("/data/PatrolAi/maskrcnn/oil_air.pth",num_classes=2) # 加载油位的maskrcnn模型

def get_input_data(cfg):
    """
    提取input_data中的信息。
    return:
        img_ref: 模板图片数据
        pointers_ref: 坐标点, 结构为{"center": [100，200], "-0.1": [200，300], "0.9": [400，500]}
        roi: 感兴趣区域, 结构为[xmin, ymin, xmax, ymax]
        number: None 或者 指针数量
        length: None 或者 指针长短
        width: None 或者 指针粗细
        color: None 或者 指针颜色
        dp: 3 或者 需要保留的小数位
    """
    img_ref = base642img(cfg["img_ref"])

    W = img_ref.shape[1]; H = img_ref.shape[0]

    ## 点坐标
    pointers = cfg["pointers"]
    pointers_ref = {}
    for coor in pointers:
        pointers_ref[coor] = [int(pointers[coor][0] * W), int(pointers[coor][1] * H)]

    if len(pointers_ref) > 3:
        ## pointers_ref头尾增加两个点
        pointers_ref = add_head_end_ps(pointers_ref)

    ## 感兴趣区域
    roi = None # 初始假设
    if "bboxes" in cfg:
        if isinstance(cfg["bboxes"], dict):
            if "roi" in cfg["bboxes"]:
                if isinstance(cfg["bboxes"]["roi"], list):
                    if len(cfg["bboxes"]["roi"]) == 4:
                        W = img_ref.shape[1]; H = img_ref.shape[0]
                        roi = cfg["bboxes"]["roi"]
                        roi = [int(roi[0]*W), int(roi[1]*H), int(roi[2]*W), int(roi[3]*H)]
    
    ## osd区域
    osd = None # 初始假设
    if "bboxes" in cfg:
        if isinstance(cfg["bboxes"], dict):
            if "osd" in cfg["bboxes"]:
                if isinstance(cfg["bboxes"]["osd"], list):
                    osd_ = cfg["bboxes"]["osd"]
                    osd=[]
                    for o_ in osd_:
                        osd.append([max(0,o_[0]-0.01),max(0,o_[1]-0.01),min(1,o_[2]+0.01),min(1,o_[3]+0.01)])
    
    ## 其他信息
    number = 1
    if "number" in cfg:
        if isinstance(cfg["number"], int):
            if cfg["number"] != -1:
                number = cfg["number"]

    length = None
    if "length" in cfg:
        if isinstance(cfg["length"], int):
            if cfg["length"] != -1:
                length = cfg["length"]
    color = None
    if "color" in cfg:
        if isinstance(cfg["color"], int):
            if cfg["color"] != -1:
                color = cfg["color"]
    width = None
    if "width" in cfg:
        if isinstance(cfg["width"], int):
            if cfg["width"] != -1:
                width = cfg["width"]
    dp = 3
    if "dp" in cfg:
        if isinstance(cfg["dp"], int):
            if cfg["dp"] != -1:
                dp = cfg["dp"]
    
    type_ = "pointer"
    if "type" in cfg:
        if isinstance(cfg["type"], str):
            type_ = cfg["type"]

    raw_val = "raw_val"
    if "raw_val" in cfg:
        if isinstance(cfg["raw_val"], str):
            raw_val = float(cfg["raw_val"])

    return img_ref, pointers_ref, roi, number, length, width, color, dp, osd, type_, raw_val

def val_dp(val, dp):
    val = round(val, dp)
    if dp == 0:
        val = int(val)
    return val

def cal_pointer(input_cfg, img_tag):
    """
    args:
        cfg: 模板信息，
            格式为，{
                "type": "pointer",
                "number": 1, 
                "dp": 3, 
                "length": 0, 
                "width": 0,
                "color": 0,
                "raw_val": 0.5,
                "pointers": {"center": [0.1, 0.2], "-0.1": [0.3, 0.4], "0.9": [0.5, 0.6], ..},
                "bboxes": {"roi":[0.1, 0.2, 0.3, 1.0], "osd": [[0.1, 0.2, 0.3, 1.0], [0.1, 0.2, 0.3, 0.4]]}
                "img_ref": "base64 image", 
                }
        img_tag: 待分析图
    return:
        val: 读数
    """
    ## 提取输入请求信息
    img_ref, pointers_ref, roi, number, length, width, color, dp, osd, type_, raw_val= get_input_data(input_cfg)
    ## 计算指针
    seg_cfgs, roi_tag = pointer_detect(img_tag, number)

    ## 如果没有
    if len(seg_cfgs) == 0:
        val = val_dp(raw_val, dp)
        return val

    ## 矫正信息
    if osd is None:
        osd = [[0,0,1,0.1],[0,0.9,1,1]]
    feat_ref = sift_create(img_ref, rm_regs=osd)
    feat_tag = sift_create(img_tag)
    M = sift_match(feat_ref, feat_tag, ratio=0.5, ops="Perspective")

    ## 求出目标图像的感兴趣区域
    if roi is not None and M is not None:
        coors = [(roi[0],roi[1]), (roi[2],roi[1]), (roi[2],roi[3]), (roi[0],roi[3])]
        coors_ = []
        for coor in coors:
            coors_.append(list(convert_coor(coor, M)))
        xs = [coor[0] for coor in coors_]
        ys = [coor[1] for coor in coors_]
        xmin = max(0, min(xs)); ymin = max(0, min(ys))
        xmax = min(img_tag.shape[1], max(xs)); ymax = min(img_tag.shape[0], max(ys))
        roi_tag = [xmin, ymin, xmax, ymax]

    ## 将不在感兴趣区域的指针筛选出去
    _seg_cfgs = []
    for cfg in seg_cfgs:
        box_ = cfg["box"]
        if is_include(box_, roi_tag, srate=0.8):
            _seg_cfgs.append(cfg)
    seg_cfgs = _seg_cfgs

    if len(seg_cfgs) == 0:
        val = val_dp(raw_val, dp)
        return val

    ## 将指针按score从大到小排列
    scores = [cfg["score"] for cfg in seg_cfgs]
    i_sort = np.argsort(np.array(scores))
    seg_cfgs = [seg_cfgs[i_sort[i]] for i in range(len(i_sort))]

    ## 筛选指针
    i = select_pointer(img_tag, seg_cfgs, number, length, width, color)
    seg = seg_cfgs[i]["seg"]

    ## 使用映射变换矫正目标图，并且转换坐标点。
    pointers_tag = conv_coor(pointers_ref, M)
        
    ## 求指针读数
    if M is not None:
        val = cal_base_scale(pointers_tag, seg)
    else:
        xo = pointers_tag["center"][0]; yo = pointers_tag["center"][1]
        if (seg[0]-xo)**2+(seg[1]-yo)**2 > (seg[2]-xo)**2+(seg[3]-yo)**2:
            seg = [seg[2], seg[3], seg[0], seg[1]]
        dx_ = seg[2] - seg[0]; dy_ = seg[3] - seg[1]
        seg_ = [xo, yo, xo + dx_, yo + dy_]
        val = cal_base_scale(pointers_tag, seg_)
        if val == None:
            seg_ = [xo, yo, xo - dx_, yo - dy_]
            val = cal_base_scale(pointers_tag, seg_)

    if val == None:
        val = val_dp(raw_val, dp)
        return val

    val = val_dp(val, dp)

    return val

def cal_counter(input_cfg, img_tag):

    ## 去cfg的参数
    img_ref, pointers_ref, roi, number, length, width, color, dp, osd, type_, raw_val= get_input_data(input_cfg)
    
    ## 求偏移矫正
    if osd is None:
        osd = [[0,0,1,0.1],[0,0.9,1,1]]
    feat_ref = sift_create(img_ref, rm_regs=osd)
    feat_tag = sift_create(img_tag)
    M = sift_match(feat_ref, feat_tag, ratio=0.5, ops="Perspective")

    ## 矫正roi框
    if M is None or roi is None:
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
    
    labels = ["0","1","2","3","4","5","6","7","8","9"]
    img_roi = img_tag[int(roi_tag[1]): int(roi_tag[3]), int(roi_tag[0]): int(roi_tag[2])]
    cfgs_roi = inference_yolov5(yolov5_ShuZiBiaoJi, img_roi, resize=640, pre_labels=labels)
    cfgs_all = inference_yolov5(yolov5_ShuZiBiaoJi, img_tag, resize=640, pre_labels=labels)

    ## 挑选最接近数量的指针为最终结果
    if len(cfgs_all) < number and len(cfgs_roi) >= number:
        true_type = "roi"
    elif len(cfgs_roi) < number and len(cfgs_all) >= number:
        true_type = "all"
    else:
        if abs(len(cfgs_roi) - number) <= abs(len(cfgs_all) - number):
            true_type = "roi"
        else:
            true_type = "all"

    if true_type == "all":
        cfgs = cfgs_all
    else:
        cfgs = cfgs_roi
    
    if len(cfgs) < 1:
        val = val_dp(raw_val, 0)
        return val

    l = [cfg["coor"][0] for cfg in cfgs]
    rank = [index for index,value in sorted(list(enumerate(l)),key=lambda x:x[1])]
    vals = [cfgs[i]["label"] for i in rank]

    val_s = ""
    for val in vals:
        val_s = val_s + val
    
    val = int(val_s)

    return val

def cal_yeweiji(input_cfg, img_tag):
    ## 去cfg的参数
    img_ref, pointers_ref, roi, number, length, width, color, dp, osd, type_, raw_val= get_input_data(input_cfg)

    ## 求偏移矫正
    if osd is None:
        osd = [[0,0,1,0.1],[0,0.9,1,1]]
    feat_ref = sift_create(img_ref, rm_regs=osd)
    feat_tag = sift_create(img_tag)
    M = sift_match(feat_ref, feat_tag, ratio=0.5, ops="Perspective")

    ## 矫正roi框
    if M is None or roi is None:
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
    
    contours, _, (masks, classes, scores) = inference_maskrcnn(maskrcnn_oil, img_roi)


    if len(masks) < 1:
        value = float(raw_val)
        return value

    ## 计算油率
    air_s = 0; oil_s = 0
    if 0 in classes:
        air_s = np.sum(masks[list(classes).index(0)])
    if 1 in classes:
        oil_s= np.sum(masks[list(classes).index(1)])
    value = oil_s / (air_s + oil_s)

    value = round(value, dp)


    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source',
        type=str,
        default='./test/pointer_test',
        help='source dir.')
    parser.add_argument(
        '--out_file',
        type=str,
        default='./yt_bj_infer.txt',
        help='out dir of saved result.')
    parser.add_argument(
        '--cfgs',
        type=str,
        default='./meter_cfgs',
        help='cfg dir')
    parser.add_argument(
        '--model_file',
        type=str,
        default="model.npy",
        help='model.npy')
    args, unparsed = parser.parse_known_args()

    test_dir = args.source # 待测试文件目录
    out_file = args.out_file # 结果保存目录
    cfg_dir = args.cfgs # md5列表目录
    model_file = args.model_file

    if os.path.exists(model_file):
        model_dict = np.load(model_file, allow_pickle=True).item()
    else:
        model_dict = {}

    fo = open(out_file, "w", encoding='utf-8')
    fo.write("ID,Name,Val\n")
    count = 0
    for json_file in glob.glob(os.path.join(cfg_dir, "*.json")):
        f = open(json_file, "r", encoding='utf-8')
        input_cfg = json.load(f)
        f.close()

        id_ = os.path.basename(json_file)[:-7]

        for img_file in glob.glob(os.path.join(test_dir, id_ + "*.jpg")):

            img_name = os.path.basename(img_file)

            if img_file.endswith("_0.jpg"):
                continue
            
            print(img_file)
            count += 1
            if img_name in model_dict:
                val = model_dict[img_name]
            else:
                img_tag = cv2.imread(img_file)
                if input_cfg["type"] == "pointer":
                    val = cal_pointer(input_cfg, img_tag)
                elif input_cfg["type"] == "counter":
                    val = cal_counter(input_cfg, img_tag)
                elif input_cfg["type"] == "yeweiji":
                    val = cal_yeweiji(input_cfg, img_tag)
                else:
                    print("data type is wrong !")
            
            s = str(count) + "," + img_name + "," + str(val)
            fo.write(s + "\n")
            print(s)
    
    fo.close()
            
