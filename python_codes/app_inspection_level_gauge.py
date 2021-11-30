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
maskrcnn_pointer = load_maskrcnn_model("/data/inspection/maskrcnn/oil_air.pth") # 加载指针的maskrcnn模型


def get_input_data(input_data):
    """
    提取input_data中的信息。
    return:
        img_tag: 目标图片数据
        img_ref: 模板图片数据
        roi: 感兴趣区域, 结构为[xmin, ymin, xmax, ymax]
        sp: 液位计的形状，圆形或方形
        dp: None 或者 指针长短
        dp: 需要保留的小数位
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
    sp = 1
    if "sp" in input_data["config"]:
        if isinstance(input_data["config"]["sp"], int):
            if input_data["config"]["sp"] != -1:
                sp = input_data["config"]["sp"]
    dp = 3
    if "dp" in input_data["config"]:
        if isinstance(input_data["config"]["dp"], int):
            if input_data["config"]["dp"] != -1:
                dp = input_data["config"]["dp"]
    
    return img_tag, img_ref, pointers_ref, roi, sp, dp


def inspection_level_gauge(input_data):

    ## 初始化输入输出信息。
    TIME_START = time.strftime("%m-%d-%H-%M-%S") 
    save_path = os.path.join("inspection_result/level_gauge", TIME_START)
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
    contours, boxes = inference_maskrcnn(maskrcnn_pointer, img_roi)
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
    f = open("/home/yh/image/python_codes/inspection_result/pointer/11-05-14-55-56/input_data.json","r", encoding='utf-8')
    input_data = json.load(f)
    f.close()
    out_data = inspection_pointer(input_data)
    print(1)


if __name__ == '__main__':
    main()
    

