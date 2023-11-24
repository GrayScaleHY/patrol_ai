import os
import cv2
import time
import json
from lib_image_ops import base642img, img2base64, img_chinese
import numpy as np
from lib_inference_yolov5 import inference_yolov5, load_yolov5_model
from lib_inference_yolov8 import load_yolov8_model, inference_yolov8
from lib_analysis_meter import angle_scale, segment2angle, angle2sclae, intersection_arc, cfgs2segs, intersection_pointers
from lib_img_registration import registration, convert_coor
from lib_help_base import color_area, GetInputData
import math

yolov5_meter = load_yolov5_model("/data/PatrolAi/yolov5/meter.pt") # 表记检测
yolov8_pointer = load_yolov8_model("/data/PatrolAi/yolov8/pointer.pt") # 指针分割

def is_include(sub_box, par_box, srate=0.8):

    sb = sub_box
    pb = par_box
    sb = [min(sb[0], sb[2]), min(sb[1], sb[3]),
          max(sb[0], sb[2]), max(sb[1], sb[3])]
    pb = [min(pb[0], pb[2]), min(pb[1], pb[3]),
          max(pb[0], pb[2]), max(pb[1], pb[3])]

    # 至少一个点在par_box里面
    points = [[sb[0], sb[1]], [sb[2], sb[1]], [sb[0], sb[3]], [sb[2], sb[3]]]
    is_in = False
    for p in points:
        if p[0] >= pb[0] and p[0] <= pb[2] and p[1] >= pb[1] and p[1] <= pb[3]:
            is_in = True
    if not is_in:
        return False

    # 判断交集占多少
    xmin = max(pb[0], sb[0])
    ymin = max(pb[1], sb[1])
    xmax = min(pb[2], sb[2])
    ymax = min(pb[3], sb[3])
    s_include = (xmax-xmin) * (ymax-ymin)
    s_box = (sb[2]-sb[0]) * (sb[3]-sb[1])
    if s_include / s_box >= srate:
        return True
    else:
        return False


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

    # 使用偏移矩阵转换坐标。
    coors_tag = {}
    for scale in coors_float:
        coor = coors_float[scale]
        coor = [coor[0] - d_ref[0], coor[1] - d_ref[1]]  # 调整坐标
        coor = convert_coor(coor, M)  # 坐标转换
        coor_tag = [coor[0] + d_tag[0], coor[1]+d_tag[1]]  # 调整回目标坐标
        coors_tag[scale] = coor_tag
    return coors_tag


def cal_base_angle(coordinates, segment):
    """
    使用角度的方法计算线段的刻度。
    args:
        coordinates: 刻度的坐标点，格式如 {"center": [398, 417], -0.1: [229, 646], 0.9: [641, 593]}
        segment: 线段，格式为 [x1, y1, x2, y2]
    """
    cfg = {}
    for scale in coordinates:
        if scale == "center":
            continue
        angle = segment2angle(coordinates["center"], coordinates[scale])
        cfg[str(angle)] = scale
    if len(cfg) < 2:
        return None
    config = [cfg]
    out_cfg = angle_scale(config)[0]
    seg_ang = segment2angle((segment[0], segment[1]), (segment[2], segment[3]))
    val = angle2sclae(out_cfg, seg_ang)
    return val

def cal_base_scale(coordinates, segment, meter_type="normal"):
    """
    使用刻度计算指针读数。
    args:
        coordinates: 刻度的坐标点，格式如 {"center": [398, 417], -0.1: [229, 646], 0.9: [641, 593]}
        segment: 线段，格式为 [x1, y1, x2, y2]
    """
    ## 如果是逆时针读数，则先将coordinates中的刻度值取反。
    if meter_type == "nszb":
        scale_list = [scale for scale in coordinates if scale != "center"]
        coordinates_ = {}
        for scale in coordinates:
            if scale == "center":
                scale_ = scale
            else:
                scale_ = -scale
            if scale != max(scale_list) and scale != min(scale_list):
                coordinates_[scale_] = coordinates[scale]
        coordinates = coordinates_

    # 根据与表盘中心的距离更正segment的头尾
    xo = coordinates["center"][0]
    yo = coordinates["center"][1]
    if (segment[0]-xo)**2+(segment[1]-yo)**2 > (segment[2]-xo)**2+(segment[3]-yo)**2:
        segment = [segment[2], segment[3], segment[0], segment[1]]

    scales = []
    for scale in coordinates:
        if scale == "center":
            continue
        scales.append(scale)
    scales.sort()
    if len(scales) < 2:
        return None
    for i in range(len(scales)-1):
        arc = coordinates["center"] + \
            coordinates[scales[i]] + coordinates[scales[i+1]]
        coor = intersection_pointers(segment, arc)
        # coor = intersection_arc(segment, arc)
        if coor is not None:
            break
    if coor is None:
        return None
    
    seg = coordinates["center"] + list(coor)
    scale_1 = scales[i]
    scale_2 = scales[i+1]
    angle_1 = segment2angle(coordinates["center"], coordinates[scale_1])
    angle_2 = segment2angle(coordinates["center"], coordinates[scale_2])
    config = [{str(angle_1): scale_1, str(angle_2): scale_2}]
    out_cfg = angle_scale(config)[0]
    seg_ang = segment2angle((seg[0], seg[1]), (seg[2], seg[3]))
    val = angle2sclae(out_cfg, seg_ang)

    # 如果是逆时针表计，则读数取反。
    if meter_type == "nszb":
        val = -val

    return val

def add_head_end_ps(pointers):
    """
    在刻度盘首位增加一个更小刻度。
    """
    psi_float = [float(i) for i in pointers if i != "center"]
    psi_str = [i for i in pointers if i != "center"]
    p_center = pointers["center"]

    # 增加一个比最小刻度更小一点的刻度
    s_min = psi_str[psi_float.index(min(psi_float))]
    p_min = pointers[s_min]
    s_min = float(s_min)
    l_min = math.sqrt((p_min[0]-p_center[0])**2 + (p_min[1]-p_center[1])**2)
    s_min_new = str(s_min - 0.0000001)
    if p_min[0] < p_center[0]:
        pointers[s_min_new] = [p_min[0], int(math.ceil(p_min[1] + l_min / 10))]
    elif p_min[0] > p_center[0]:
        pointers[s_min_new] = [p_min[0], int(math.ceil(p_min[1] - l_min / 10))]
    else:
        if p_min[1] > p_center[1]:
            pointers[s_min_new] = [
                int(math.ceil(p_min[0] + l_min / 10)), p_min[1]]
        else:
            pointers[s_min_new] = [
                int(math.ceil(p_min[0] - l_min / 10)), p_min[1]]

    # 增加一个比最大刻度更大一点的刻度
    s_max = psi_str[psi_float.index(max(psi_float))]
    p_max = pointers[s_max]
    s_max = float(s_max)
    l_max = math.sqrt((p_max[0]-p_center[0])**2 + (p_max[1]-p_center[1])**2)
    s_max_new = str(s_max + 0.0000001)
    if p_max[0] > p_center[0]:
        pointers[s_max_new] = [p_max[0], int(math.ceil(p_max[1] + l_max / 10))]
    elif p_max[0] < p_center[0]:
        pointers[s_max_new] = [p_max[0], int(math.ceil(p_max[1] - l_max / 10))]
    else:
        if p_max[1] > p_center[1]:
            pointers[s_max_new] = [
                int(math.ceil(p_max[0] - l_max / 10)), p_max[1]]
        else:
            pointers[s_max_new] = [
                int(math.ceil(p_max[0] + l_max / 10)), p_max[1]]

    return pointers


def select_pointer(img, seg_cfgs, length, width, color):
    """
    根据指针长短，粗细，颜色来筛选指针
    返回index
    """
    if len(seg_cfgs) == 1:
        return 0

    # seg_cfgs = seg_cfgs[:int(number)]
    segs = []
    for cfg in seg_cfgs:
        s = cfg["seg"]
        seg = [min(s[0], s[2]), min(s[1], s[3]),
               max(s[0], s[2]), max(s[1], s[3])]
        segs.append(seg)

    if length is not None:
        lengths = [(a[2]-a[0])**2 + (a[3]-a[1])**2 for a in segs]
        if length == 0:
            return lengths.index(min(lengths))
        elif length == 2:
            return lengths.index(max(lengths))
        else:
            lengths.pop(lengths.index(min(lengths)))
            return lengths.index(min(lengths))

    elif width is not None:
        widths = [min(a[2]-a[0], a[3]-a[1]) for a in segs]
        if length == 0:
            return widths.index(min(widths))
        elif length == 2:
            return widths.index(max(widths))
        else:
            widths.pop(widths.index(min(widths)))
            return widths.index(min(widths))

    elif color is not None:
        if len(img.shape) == 2:
            return 0
        area_max = 0
        i_max = 0
        for i in range(len(seg_cfgs)):
            _img = img.copy()
            mask = seg_cfgs[i]["mask"]

            # 将mask外的区域填充为蓝色
            for j in range(img.shape[-1]):
                if j != 0:
                    _img[:, :, j] = _img[:, :, j] * mask
                else:
                    _img[:, :, j] = _img[:, :, j] * mask + (mask - 1)
            color_list = ["black", "white", "red", "red2"]
            color_ = color_area(_img, color_list)
            print(seg_cfgs[i]["seg"])
            print(color_)
            if int(color) == 0:
                c_area = (color_["black"] + 1) / \
                    (sum([color_[_c] for _c in color_list]) + 1)
            elif int(color) == 1:
                c_area = (color_["white"] + 1) / \
                    (sum([color_[_c] for _c in color_list]) + 1)
            elif int(color) == 2:
                c_area = (color_["red"] + color_["red2"] + 1) / \
                    (sum([color_[_c] for _c in color_list]) + 1)
            if c_area > area_max:
                area_max = c_area
                i_max = i

        return i_max
    else:
        return 0


def pointer_detect(img_tag, number):
    """
    args:
        img_tag: 图片
    return:
        seg_cfgs: 指针信息, 格式为[{"seg": seg, "box": box, "score": score}, ..]
        roi_tag: 感兴趣区域
    """
    # 识别图中的表盘
    h, w = img_tag.shape[:2]
    cfgs = inference_yolov5(yolov5_meter, img_tag, resize=640)
    bboxes = [[0, 0, w, h]] + [cfg["coor"] for cfg in cfgs]

    # 找到bboxes中的所有指针
    seg_cfgs_all = []  # 整张图来推理时检测的指针信息
    seg_cfgs_part = []  # 将表记部分抠出来检测的指针信息
    for j, c in enumerate(bboxes):
        img = img_tag[c[1]:c[3], c[0]:c[2]]

        cfgs = inference_yolov8(yolov8_pointer,img, same_iou_thres=0.85) # 指针分割推理
        cfgs = cfgs2segs(cfgs) 

        for i in range(len(cfgs)):

            ## 指针的mask
            mask = cfgs[i]["mask"]
            mask_raw = np.zeros(img_tag.shape[:2], dtype=np.uint8)
            mask_raw[c[1]:c[3], c[0]:c[2]] = mask
            s = cfgs[i]["seg"]
            score = cfgs[i]["score"]
            b = cfgs[i]["coor"]
            box = [b[0]+c[0], b[1]+c[1], b[2]+c[0], b[3]+c[1]]
            seg = [s[0]+c[0], s[1]+c[1], s[2]+c[0], s[3]+c[1]]
            cfg = {"seg": seg, "box": box, "score": score, "mask": mask_raw}
            if j == 0:
                seg_cfgs_all.append(cfg)
            else:
                seg_cfgs_part.append(cfg)
    

    # 挑选最接近数量的指针为最终结果
    if len(seg_cfgs_all) >= number:
        true_type = "all"
    elif len(seg_cfgs_all) < number and len(seg_cfgs_part) >= number:
        true_type = "part"
    else:
        if abs(len(seg_cfgs_part) - number) < abs(len(seg_cfgs_all) - number):
            true_type = "part"
        else:
            true_type = "all"

    if true_type == "all":
        seg_cfgs = seg_cfgs_all
        roi_tag = bboxes[0]
    else:
        seg_cfgs = seg_cfgs_part
        b = np.array(bboxes[1:], dtype=int)
        roi_tag = [min(b[:, 0]), min(b[:, 1]), max(b[:, 2]), max(b[:, 3])]

    return seg_cfgs, roi_tag


def segs2val(img_tag, pointers_tag, seg_cfgs, length, width, color, val_size, meter_type):
    """
    根据seg_cfgs求指针读数
    """
    if len(seg_cfgs) == 0:
        return [], None
    
    ## 通过val_size读数大小来筛选指针。
    if val_size is not None:
        vals = []
        seg_cfgs_real = []
        for seg_cfg in seg_cfgs:
            seg = seg_cfg["seg"]
            val = cal_base_scale(pointers_tag, seg, meter_type)
            if val != None:
                vals.append(val)
                seg_cfgs_real.append(seg_cfg)
                
        if len(vals) < 1:
            return [], None
        
        i_min = vals.index(min(vals))
        i_max = vals.index(max(vals))
        if len(seg_cfgs_real) < 3:
            i_mid = i_min
        else:
            i_mid = [i for i in range(len(seg_cfgs_real)) if i != i_min and i != i_max][0]

        if val_size == 0:
            seg = seg_cfgs_real[i_min]["seg"]; val = vals[i_min]
        elif val_size == 2:
            seg = seg_cfgs_real[i_max]["seg"]; val = vals[i_max]
        else:
            seg = seg_cfgs_real[i_mid]["seg"]; val = vals[i_mid]
        return [seg], val
    
    ## 双指针动作次数表
    if meter_type == "blq_zzscsb" or meter_type == "nszb_zzscsb":
        if meter_type == "nszb_zzscsb":
            meter_type = "nszb"

        i = select_pointer(img_tag, seg_cfgs, 2, width, color)
        seg1 = seg_cfgs[i]["seg"]
        val1 = cal_base_scale(pointers_tag, seg1, meter_type)
        i = select_pointer(img_tag, seg_cfgs, 0, width, color)
        seg2 = seg_cfgs[i]["seg"]
        val2 = cal_base_scale(pointers_tag, seg2, meter_type)
        if val1 == None and val2 == None:
            return [], None
        if val1 == None:
            val1 = val2
        if val2 == None:
            val2 = val1
        val = round(val2) * 10 + round(val1)
        return [seg1, seg2], val

    i = select_pointer(img_tag, seg_cfgs, length, width, color)
    segs = [seg_cfgs[i]["seg"]]

    ## 求指针读数
    val = cal_base_scale(pointers_tag, segs[0], meter_type)

    return segs, val

def inspection_pointer(input_data):

    # 提取输入请求信息
    DATA = GetInputData(input_data)
    checkpoint = DATA.checkpoint
    an_type = DATA.type
    img_tag = DATA.img_tag
    img_ref = DATA.img_ref
    pointers_ref = DATA.pointers
    number = DATA.number
    roi = DATA.roi
    osd = DATA.osd
    dp = DATA.dp
    length = DATA.length
    width = DATA.width
    color = DATA.color
    val_size = DATA.val_size
    meter_type = DATA.meter_type

    # 刻度点左右添加两个点。
    pointers_ref = add_head_end_ps(pointers_ref)

    # 初始化输出结果
    out_data = {"code": 0, "data": {
    }, "img_result": input_data["image"], "msg": "Request " + an_type + ";"}  # 初始化输出信息

    # 画上点位名称和osd区域
    img_tag_ = img_tag.copy()
    img_tag_ = img_chinese(img_tag_, an_type + "_" + checkpoint, (10, 100), color=(255, 0, 0), size=30)
    for o_ in osd:  # 如果配置了感兴趣区域，则画出osd区域
        cv2.rectangle(img_tag_, (int(o_[0]), int(o_[1])), (int(
            o_[2]), int(o_[3])), (255, 0, 255), thickness=1)
        cv2.putText(img_tag_, "osd", (int(o_[0]), int(
            o_[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)

    if an_type != "pointer":
        out_data["msg"] = out_data["msg"] + "type isn't pointer; "
        out_data["code"] = 1
        img_tag_ = img_chinese(
            img_tag_, out_data["msg"], (10, 130), color=(255, 0, 0), size=30)
        out_data["img_result"] = img2base64(img_tag_)
        return out_data

    # 检测指针
    seg_cfgs, roi_tag = pointer_detect(img_tag, number)

    if len(seg_cfgs) == 0:
        out_data["msg"] = out_data["msg"] + "Can not find pointer in image; "
        out_data["code"] = 1
        img_tag_ = img_chinese(
            img_tag_, out_data["msg"], (10, 130), color=(255, 0, 0), size=30)
        out_data["img_result"] = img2base64(img_tag_)
        return out_data

    # 画出所有指针
    for cfg in seg_cfgs:
        seg = cfg["seg"]
        cv2.line(img_tag_, (int(seg[0]), int(seg[1])),
                 (int(seg[2]), int(seg[3])), (255, 0, 255), 1)

    # 求偏移矩阵
    M = registration(img_ref, img_tag)
    
    # 求出测试图的感兴趣区域
    if len(roi) > 0 and M is not None:
        roi = roi[0]
        coors = [(roi[0], roi[1]), (roi[2], roi[1]),
                 (roi[2], roi[3]), (roi[0], roi[3])]
        coors_ = [list(convert_coor(coor, M)) for coor in coors]
        c_ = np.array(coors_, dtype=int)
        H, W = img_tag.shape[:2]
        r = [min(c_[:, 0]), min(c_[:, 1]), max(c_[:, 0]), max(c_[:, 1])]
        roi_tag = [max(0, r[0]), max(0, r[1]), min(W, r[2]), min(H, r[3])]

    # 画出roi_tag
    c = roi_tag
    cv2.rectangle(img_tag_, (int(c[0]), int(c[1])), (int(
        c[2]), int(c[3])), (255, 0, 0), thickness=1)
    cv2.putText(img_tag_, "roi", (int(c[0]), int(
        c[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)

    # 保留在roi_tag中的指针
    seg_cfgs = [cfg for cfg in seg_cfgs if is_include(
        cfg["box"], roi_tag, srate=0.8)]

    if len(seg_cfgs) == 0:
        out_data["msg"] = out_data["msg"] + "Can not find pointer in roi; "
        out_data["code"] = 1
        img_tag_ = img_chinese(
            img_tag_, out_data["msg"], (10, 130), color=(255, 0, 0), size=30)
        out_data["img_result"] = img2base64(img_tag_)
        return out_data

    # 使用映射变换矫正目标图，并且转换坐标点。
    pointers_tag = conv_coor(pointers_ref, M)
    for scale in pointers_tag:
        coor = pointers_tag[scale]
        cv2.circle(img_tag_, (int(coor[0]), int(coor[1])), 1, (255, 0, 255), 8)
        cv2.putText(img_tag_, str(scale), (int(coor[0]), int(
            coor[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)

    if "center" not in pointers_tag:
        out_data["msg"] = out_data["msg"] + \
            "Can not find center in pointers_tag; "
        out_data["code"] = 1
        out_data["img_result"] = img2base64(img_tag_)
        img_tag_ = img_chinese(
            img_tag_, out_data["msg"], (10, 130), color=(255, 0, 0), size=30)
        return out_data

    # 将指针按score从大到小排列，筛选指针
    scores = [cfg["score"] for cfg in seg_cfgs]
    i_sort = np.argsort(np.array(scores))
    seg_cfgs = [seg_cfgs[i_sort[i]] for i in range(len(i_sort))]  # 排序

    # 根据seg_cfgs求val
    segs, val = segs2val(img_tag, pointers_tag, seg_cfgs, length, width, color, val_size, meter_type)
    for seg in segs:
        cv2.line(img_tag_, (int(seg[0]), int(seg[1])),(int(seg[2]), int(seg[3])), (0, 255, 0), 2)

    if val == None:
        out_data["msg"] = out_data["msg"] + "Can not find ture pointer; "
        out_data["code"] = 1
        out_data["img_result"] = img2base64(img_tag_)
        img_tag_ = img_chinese(
            img_tag_, out_data["msg"], (10, 130), color=(255, 0, 0), size=30)
        return out_data

    if meter_type == "blq_zzscsb" or meter_type == "nszb_zzscsb":
        dp = 0
    if dp == 0:
        val = int(round(val, dp))
    else:
        val = round(val, dp)
    seg = [float(seg[0]), float(seg[1]), float(seg[2]), float(seg[3])]
    roi_tag = [float(roi_tag[0]), float(roi_tag[1]),
               float(roi_tag[2]), float(roi_tag[3])]
    out_data["data"] = {"type": "pointer",
                        "values": val, "segment": seg, "bbox": roi_tag}

    # 画出指针读数
    H, W = img_tag.shape[:2]
    s = W / 800
    cv2.putText(img_tag_, str(val), (int(seg[2]), int(
        seg[3])-5), cv2.FONT_HERSHEY_SIMPLEX, math.ceil(s), (0, 255, 0), thickness=math.ceil(s*2))
    img_tag_ = img_chinese(
        img_tag_, out_data["msg"], (10, 130), color=(255, 0, 0), size=30)
    out_data["img_result"] = img2base64(img_tag_)

    return out_data


if __name__ == '__main__':
    import glob
    from lib_help_base import save_input_data, save_output_data, get_save_head
    # # for img_tag_file in glob.glob("12-06-11-48-55_img_tag*.jpg"):
    # img_tag_file = "/data/PatrolAi/patrol_ai/python_codes/test/test_img/边缘_绕阻温度表1.png"
    # img_ref_file = "/data/PatrolAi/patrol_ai/python_codes/test/test_img/边缘_绕阻温度表1.png"
    # pointers = {"center": [1483, 410],
    #             "0": [1271, 610],
    #             "20": [1203, 381],
    #             "40": [1351, 155],
    #             "80": [1713, 381],
    #             "100": [1639, 523]}
    # # bboxes = {"roi": [805, 256, 1217, 556]}
    # img_ref = cv2.imread(img_ref_file)
    # W = img_ref.shape[1]
    # H = img_ref.shape[0]
    # for t in pointers:
    #     pointers[t] = [pointers[t][0]/W, pointers[t][1]/H]
    # # for b in bboxes:
    # #     bboxes[b] = [bboxes[b][0]/W, bboxes[b][1]/H, bboxes[b][2]/W, bboxes[b][3]/H]

    # img_tag = img2base64(cv2.imread(img_tag_file))
    # img_ref = img2base64(cv2.imread(img_ref_file))
    # config = {
    #     "img_ref": img_ref,
    #     "number": 2,
    #     "pointers": pointers,
    #     "color": 2
    # }
    # input_data = {"image": img_tag, "config": config, "type": "pointer"}
    json_file = "/data/PatrolAi/result_patrol/1123084815__input_data.json"
    print(json_file)
    f = open(json_file,"r", encoding='utf-8')
    input_data = json.load(f)
    f.close()
    out_data = inspection_pointer(input_data)
    save_dir, name_head = get_save_head(input_data)
    save_input_data(input_data, save_dir, name_head, draw_img=True)
    save_output_data(out_data, save_dir, name_head)
    print("------------------------------")
    for s in out_data:
        if s != "img_result":
            print(s, ":", out_data[s])
    print("------------------------------")
