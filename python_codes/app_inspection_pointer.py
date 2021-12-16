import os
import cv2
import time
import json
from lib_image_ops import base642img, img2base64, img_chinese
import numpy as np
from lib_inference_yolov5 import load_yolov5_model, inference_yolov5
from lib_analysis_meter import angle_scale, segment2angle, angle2sclae, draw_result
from lib_inference_mrcnn import load_maskrcnn_model, inference_maskrcnn, contour2segment, intersection_arc
from app_inspection_disconnector import sift_match, convert_coor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## 加载模型
yolov5_meter = load_yolov5_model("/data/inspection/yolov5/meter.pt") # 加载仪表yolov5模型
maskrcnn_pointer = load_maskrcnn_model("/data/inspection/maskrcnn/pointer.pth") # 加载指针的maskrcnn模型


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


def cal_base_scale(coordinates, segment):
    """
    使用刻度计算指针读数。
    args:
        coordinates: 刻度的坐标点，格式如 {"center": [398, 417], -0.1: [229, 646], 0.9: [641, 593]}
        segment: 线段，格式为 [x1, y1, x2, y2]
    """
    scales = []
    for scale in coordinates:
        if scale == "center":
            continue
        scales.append(scale)
    scales.sort()
    if len(scales) < 2:
        return None
    for i in range(len(scales)-1):
        arc = coordinates["center"] + coordinates[scales[i]] + coordinates[scales[i+1]]
        coor = intersection_arc(segment, arc)
        if coor is not None:
            break
    if coor is None:
        return None
    seg = coordinates["center"] + list(coor)
    scale_1 = scales[i]
    scale_2 = scales[i+1]
    angle_1 = segment2angle(coordinates["center"],coordinates[scale_1])
    angle_2 = segment2angle(coordinates["center"],coordinates[scale_2])
    config = [{str(angle_1): scale_1, str(angle_2): scale_2}]
    out_cfg = angle_scale(config)[0]
    seg_ang = segment2angle((seg[0], seg[1]), (seg[2], seg[3]))
    val = angle2sclae(out_cfg, seg_ang)
    return val


def get_input_data(input_data):
    """
    提取input_data中的信息。
    return:
        img_tag: 目标图片数据
        img_ref: 模板图片数据
        pointers_ref: 坐标点, 结构为{"center": [100，200], "-0.1": [200，300], "0.9": [400，500]}
        roi: 感兴趣区域, 结构为[xmin, ymin, xmax, ymax]
        number: None 或者 指针数量
        length: None 或者 指针长短
        width: None 或者 指针粗细
        color: None 或者 指针颜色
        dp: 3 或者 需要保留的小数位
    """
    img_tag = base642img(input_data["image"])
    img_ref = base642img(input_data["config"]["img_ref"])

    W = img_ref.shape[1]; H = img_ref.shape[0]

    ## 点坐标
    pointers = input_data["config"]["pointers"]
    pointers_ref = {}
    for coor in pointers:
        pointers_ref[coor] = [int(pointers[coor][0] * W), int(pointers[coor][1] * H)]

    ## 感兴趣区域
    roi = None # 初始假设
    if "bboxes" in input_data["config"]:
        if isinstance(input_data["config"]["bboxes"], dict):
            if "roi" in input_data["config"]["bboxes"]:
                if isinstance(input_data["config"]["bboxes"]["roi"], list):
                    if len(input_data["config"]["bboxes"]["roi"]) == 4:
                        W = img_ref.shape[1]; H = img_ref.shape[0]
                        roi = input_data["config"]["bboxes"]["roi"]
                        roi = [int(roi[0]*W), int(roi[1]*H), int(roi[2]*W), int(roi[3]*H)]
    
    ## 其他信息
    number = None
    if "number" in input_data["config"]:
        if isinstance(input_data["config"]["number"], int):
            if input_data["config"]["number"] != -1:
                number = input_data["config"]["number"]
    length = None
    if "length" in input_data["config"]:
        if isinstance(input_data["config"]["length"], int):
            if input_data["config"]["length"] != -1:
                length = input_data["config"]["length"]
    color = None
    if "color" in input_data["config"]:
        if isinstance(input_data["config"]["color"], int):
            if input_data["config"]["color"] != -1:
                color = input_data["config"]["color"]
    width = None
    if "width" in input_data["config"]:
        if isinstance(input_data["config"]["width"], int):
            if input_data["config"]["width"] != -1:
                width = input_data["config"]["width"]
    dp = 3
    if "dp" in input_data["config"]:
        if isinstance(input_data["config"]["dp"], int):
            if input_data["config"]["dp"] != -1:
                dp = input_data["config"]["dp"]
    
    return img_tag, img_ref, pointers_ref, roi, number, length, width, color, dp

def select_pointer(img, bboxes, number, length, width, color):
    """
    根据指针长短，粗细，颜色来筛选指针
    返回index
    """
    if number is None or number == 1 or len(bboxes) <= 1:
        return 0
    bboxes = bboxes[:int(number)]
    if length is not None:
        lengths = [(a[2]-a[1])**2 + (a[3]-a[0])**2 for a in bboxes]
        if length == 0:
            return lengths.index(min(lengths))
        elif length == 2:
            return lengths.index(max(lengths))
        else:
            lengths.pop(lengths.index(min(lengths)))
            return lengths.index(min(lengths))
    elif width is not None:
        widths = [min(a[2]-a[1], a[3]-a[0]) for a in bboxes]
        if length == 0:
            return widths.index(min(widths))
        elif length == 2:
            return widths.index(max(widths))
        else:
            widths.pop(widths.index(min(widths)))
            return widths.index(min(widths))
    elif color is not None:
        color_list = [[0,0,0],[255,255,255],[0,0,255]] # 黑，白，红
        c_ref = np.array(color_list[color], dtype=float)
        d_list = []
        for b in bboxes:
            c_tag = np.array(img[int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2), :], dtype=float)
            d_list.append(np.linalg.norm(c_ref - c_tag))
        return d_list.index(min(d_list))
    else:
        return 0
    

def inspection_pointer(input_data):

    ## 初始化输入输出信息。
    TIME_START = time.strftime("%m-%d-%H-%M-%S") 
    save_path = os.path.join("inspection_result/pointer", TIME_START)
    os.makedirs(save_path, exist_ok=True)
    f = open(os.path.join(save_path, "input_data.json"), "w")
    json.dump(input_data, f, ensure_ascii=False)  # 保存输入信息json文件
    f.close()

    out_data = {"code":0, "data":[], "img_result": "image", "msg": "request sucdess; "} #初始化输出信息

    if input_data["type"] != "pointer":
        out_data["msg"] = out_data["msg"] + "type isn't pointer; "
        return out_data
    
    ## 提取输入请求信息
    img_tag, img_ref, pointers_ref, roi, number, length, width, color, dp= get_input_data(input_data)

    ## 将输入请求信息可视化
    img_tag_ = img_tag.copy()
    img_ref_ = img_ref.copy()
    cv2.imwrite(os.path.join(save_path, "img_tag.jpg"), img_tag_)
    cv2.imwrite(os.path.join(save_path, "img_ref.jpg"), img_ref_)
    for scale in pointers_ref:  # 将坐标点标注在图片上
        coor = pointers_ref[scale]
        cv2.circle(img_ref_, (int(coor[0]), int(coor[1])), 4, (255, 0, 255), 8)
        cv2.putText(img_ref_, str(scale), (int(coor[0])-5, int(coor[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), thickness=4)
    if roi is not None:   ## 如果配置了感兴趣区域，则画出感兴趣区域
        cv2.rectangle(img_ref_, (int(roi[0]), int(roi[1])),
                    (int(roi[2]), int(roi[3])), (0, 0, 255), thickness=2)
        cv2.putText(img_ref_, "roi", (int(roi[0])-5, int(roi[1])-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
    cv2.imwrite(os.path.join(save_path, "img_ref_cfg.jpg"), img_ref_)

    ## 如果没有配置roi，则自动识别表盘作为roi
    if roi is None:
        bbox_ref = inference_yolov5(yolov5_meter, img_ref, resize=640)
        if len(bbox_ref) > 0:
            roi = bbox_ref[0]["coor"]
        else:
            roi = [0,0, img_ref.shape[1], img_ref.shape[0]]
        
    M = sift_match(img_ref, img_tag, ratio=0.5, ops="Perspective")
    
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
    pointers_tag = conv_coor(pointers_ref, M)

    ## 将矫正偏移的信息写到图片中
    s = (roi_tag[2] - roi_tag[0]) / 400 # 根据框子大小决定字号和线条粗细。
    cv2.rectangle(img_tag_, (int(roi_tag[0]), int(roi_tag[1])),
                    (int(roi_tag[2]), int(roi_tag[3])), (0, 0, 255), thickness=round(s*2))
    cv2.putText(img_tag_, "meter", (int(roi_tag[0]), int(roi_tag[1]-s)),
                    cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 255), thickness=round(s)*2)
    # img_chinese(img_tag_, "表记", (int(roi_tag[0]), int(roi_tag[1])), color=(0, 0, 255), size=20)
    for scale in pointers_tag:
        coor = pointers_tag[scale]
        cv2.circle(img_tag_, (int(coor[0]), int(coor[1])), round(s/4), (255, 0, 255), 8)
        cv2.putText(img_tag_, str(scale), (int(coor[0])-5, int(coor[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, s/2, (255, 0, 255), thickness=round(s)) #round(s/4)
        

    # 用maskrcnn检测指针轮廓并且拟合成线段.
    contours, boxes, _ = inference_maskrcnn(maskrcnn_pointer, img_roi)
    segments = contour2segment(contours, boxes)

    if len(segments) == 0:
        out_data["msg"] = out_data["msg"] + "Can not find pointer; "
        cv2.imwrite(os.path.join(save_path, "img_tag_cfg.jpg"), img_tag_)

        return out_data

    ## 筛选指针
    i = select_pointer(roi_tag, boxes, number, length, width, color)
    seg = segments[i]

    ## 根据与表盘中心的距离更正segment的头尾
    xo = img_roi.shape[1] / 2; yo = img_roi.shape[0] / 2
    if (seg[0]-xo)**2+(seg[1]-yo)**2 > (seg[2]-xo)**2+(seg[3]-yo)**2:
        seg = [seg[2], seg[3], seg[0], seg[1]]

    ## 将segments还原到原图的坐标
    dx = roi_tag[0]; dy = roi_tag[1]
    seg = [seg[0]+dx, seg[1]+dy, seg[2]+dx, seg[3]+dy]

    if M is not None:
        val = cal_base_scale(pointers_tag, seg)
    else:
        val = cal_base_angle(pointers_tag, seg)

    if val == None:
        out_data["msg"] = out_data["msg"] + "Can not find ture pointer; "
        f = open(os.path.join(save_path, "output_data.json"), "w", encoding='utf-8')
        json.dump(out_data, f, indent=2, ensure_ascii=False)
        f.close()
        return out_data

    val = round(val, dp)
    seg = [float(seg[0]), float(seg[1]), float(seg[2]), float(seg[3])]
    roi_tag = [float(roi_tag[0]), float(roi_tag[1]), float(roi_tag[2]), float(roi_tag[3])]
    out_data["data"] = {"type": "pointer", "values": val, "segment": seg, "bbox": roi_tag}
    

    ## 可视化最终计算结果
    cv2.line(img_tag_, (int(seg[0]), int(seg[1])), (int(seg[2]), int(seg[3])), (0, 255, 0), round(s))
    cv2.putText(img_tag_, str(val), (int(seg[2])-5, int(seg[3])),
                cv2.FONT_HERSHEY_SIMPLEX, round(s), (0, 255, 0), thickness=round(s*2))
    # cv2.line(img_tag_, (int(seg[0]), int(seg[1])), (int(seg[2]), int(seg[3])), (0, 255, 0), 2)
    # cv2.putText(img_tag_, str(val), (int(seg[2])-5, int(seg[3])),
    #             cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), thickness=4)
    
    cv2.imwrite(os.path.join(save_path, "img_tag_cfg.jpg"), img_tag_)

    ## 输出可视化结果的图片。
    f = open(os.path.join(save_path, "output_data.json"), "w", encoding='utf-8')
    json.dump(out_data, f, indent=2, ensure_ascii=False)
    f.close()
    out_data["img_result"] = img2base64(img_tag_)

    return out_data

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # img_ref_file = "images/img_tag.jpg"
    # img_tag_file = "images/img_ref.jpg"
    # pointers ={"center": [969, 551],
    #       "-0.1": [872, 834],
    #       "0": [758, 755],
    #       "0.2": [687, 510],
    #       "0.4": [846, 310],
    #       "0.6": [1095, 309],
    #       "0.8": [1253, 505],
    #       "0.9": [1248, 642]}
    # bboxes = {"roi": [805, 256, 1217, 556]}
    # img_ref = cv2.imread(img_ref_file)
    # W = img_ref.shape[1]; H = img_ref.shape[0]
    # for t in pointers:
    #     pointers[t] = [pointers[t][0]/W, pointers[t][1]/H]
    # for b in bboxes:
    #     bboxes[b] = [bboxes[b][0]/W, bboxes[b][1]/H, bboxes[b][2]/W, bboxes[b][3]/H]
    
    # img_tag = img2base64(cv2.imread(img_tag_file))
    # img_ref = img2base64(cv2.imread(img_ref_file))
    # config = {
    #     "img_ref": img_ref, 
    #     "number": 1, 
    #     "pointers": pointers
    #     # "length": 0, 
    #     # "width": 0, 
    #     # "color": 0, 
    #     # "bboxes": bboxes
    # }
    # input_data = {"image": img_tag, "config": config, "type": "pointer"}
    f = open("/home/yh/image/python_codes/inspection_result/pointer/12-15-13-57-12/input_data.json","r", encoding='utf-8')
    input_data = json.load(f)
    f.close()
    out_data = inspection_pointer(input_data)
    print(1)


if __name__ == '__main__':
    main()
    

