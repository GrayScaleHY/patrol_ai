import os
import time
import cv2
import json
from lib_image_ops import base642img, img2base64, img_chinese
from lib_inference_yolov5 import load_yolov5_model, inference_yolov5
from lib_help_base import color_list
from app_inspection_disconnector import sift_match, convert_coor
import config_object_name
import numpy as np


def get_input_data(input_data):
    """
    提取input_data中的信息。
    return:
        img_tag: 目标图片数据
        img_ref: 模板图片数据
        roi: 感兴趣区域, 结构为[xmin, ymin, xmax, ymax]
    """

    img_tag = base642img(input_data["image"])

    ## 是否有模板图
    if "img_ref" not in input_data["config"] or input_data["config"]["img_ref"] == -1:
        img_ref = None
    else:
        img_ref = base642img(input_data["config"]["img_ref"])
        

    ## 感兴趣区域
    if "bboxes" not in input_data["config"] or input_data["config"]["bboxes"] == -1:
        roi = None
    else:
        W = img_ref.shape[1]; H = img_ref.shape[0]
        roi = input_data["config"]["bboxes"]["roi"]
        roi = [int(roi[0]*W), int(roi[1]*H), int(roi[2]*W), int(roi[3]*H)]
        
    return img_tag, img_ref, roi


yolov5_meter = load_yolov5_model("/data/inspection/yolov5/meter.pt") # 表盘
yolov5_air_switch = load_yolov5_model("/data/inspection/yolov5/air_switch.pt") # 空气开关
yolov5_fire_smoke = load_yolov5_model("/data/inspection/yolov5/fire_smoke.pt") # 烟火
yolov5_led = load_yolov5_model("/data/inspection/yolov5/led.pt") # led灯
yolov5_pressplate = load_yolov5_model("/data/inspection/yolov5/pressplate.pt") # 压板
yolov5_helmet = load_yolov5_model("/data/inspection/yolov5/helmet.pt") # 安全帽
yolov5_fanpaiqi = load_yolov5_model("/data/inspection/yolov5/fanpaiqi.pt") # 翻拍器
yolov5_rotary_switch = load_yolov5_model("/data/inspection/yolov5/rotary_switch.pt") # 切换把手(旋钮开关)

def inspection_object_detection(input_data):
    """
    yolov5的目标检测推理。
    """

    ## 初始化输入输出信息。
    img_tag, img_ref, roi = get_input_data(input_data)
    out_data = {"code": 0, "data":[], "img_result": "image", "msg": "Success request object detect; "} # 初始化out_data

    ## 选择模型
    if input_data["type"] == "pressplate": # ["air_switch", "fire_smoke", "led", "pressplate"]:
        yolov5_model = yolov5_pressplate
    elif input_data["type"] == "meter":
        yolov5_model = yolov5_meter
    elif input_data["type"] == "air_switch":
        yolov5_model = yolov5_air_switch
    elif input_data["type"] == "fire_smoke":
        yolov5_model = yolov5_fire_smoke
    elif input_data["type"] == "led":
        yolov5_model = yolov5_led
    elif input_data["type"] == "helmet":
        yolov5_model = yolov5_helmet
    elif input_data["type"] == "fanpaiqi":
        yolov5_model = yolov5_fanpaiqi
    elif input_data["type"] == "rotary_switch":
        yolov5_model = yolov5_rotary_switch
    elif input_data["type"] == "arrow":
        yolov5_model = yolov5_rotary_switch
    else:
        out_data["msg"] = out_data["msg"] + "Type isn't object; "
        return out_data

    ## 将输入请求信息可视化
    img_tag_ = img_tag.copy()
    TIME_START = time.strftime("%m-%d-%H-%M-%S") 
    save_path = os.path.join("inspection_result", input_data["type"], TIME_START)
    os.makedirs(save_path, exist_ok=True)
    f = open(os.path.join(save_path, "input_data.json"), "w")
    json.dump(input_data, f, ensure_ascii=False)  # 保存输入信息json文件
    f.close()
    cv2.imwrite(os.path.join(save_path, "img_tag.jpg"), img_tag) # 将输入图片可视化
    if img_ref is not None:
        cv2.imwrite(os.path.join(save_path, "img_ref.jpg"), img_ref) # 将输入图片可视化
    if roi is not None:   ## 如果配置了感兴趣区域，则画出感兴趣区域
        img_ref_ = img_ref.copy()
        cv2.rectangle(img_ref_, (int(roi[0]), int(roi[1])),
                    (int(roi[2]), int(roi[3])), (0, 0, 255), thickness=2)
        cv2.putText(img_ref_, "roi", (int(roi[0])-5, int(roi[1])-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
        cv2.imwrite(os.path.join(save_path, "img_ref_cfg.jpg"), img_ref_)

    ## 求出目标图像的感兴趣区域
    if roi is None:
        M = None
    else:
        M = sift_match(img_ref, img_tag, ratio=0.5, ops="Perspective")

    if M is None:
        roi_tag = [0,0, img_tag.shape[1], img_tag.shape[0]]
    else:
        coors = [(roi[0],roi[1]), (roi[2],roi[1]), (roi[2],roi[3]), (roi[0],roi[3])]
        coors_ = []
        for coor in coors:
            coors_.append(list(convert_coor(coor, M)))
        coors_ = np.array(coors_, dtype=int)
        roi_tag = [np.min(coors_[:,0]), np.min(coors_[:,1]), np.max(coors_[:,0]), np.max(coors_[:,1])]
    img_roi = img_tag[int(roi_tag[1]): int(roi_tag[3]), int(roi_tag[0]): int(roi_tag[2])]

    ## 生成目标检测信息
    boxes = inference_yolov5(yolov5_model, img_roi, resize=640) # inference
    if len(boxes) == 0: #没有检测到目标
        out_data["msg"] = out_data["msg"] + "; Not find object"
        return out_data

    ## 将bboxes映射到原图坐标
    bboxes = []
    for bbox in boxes:
        c = bbox["coor"]; r = roi_tag
        coor = [c[0]+r[0], c[1]+r[1], c[2]+r[0], c[3]+r[1]]
        bboxes.append({"label": bbox["label"], "coor": coor, "score": bbox["score"]})

    for bbox in bboxes:
        cfg = {"type": bbox["label"], "bbox": [c[0]+r[0], c[1]+r[1], c[2]+r[0], c[3]+r[1]]}
        out_data["data"].append(cfg)
    
    ## labels 和 color的对应关系
    labels = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
    colors = color_list(len(labels))
    color_dict = {}
    for i, label in enumerate(labels):
        color_dict[label] = colors[i]

    ## 可视化计算结果
    f = open(os.path.join(save_path, "out_data.json"), "w")
    json.dump(out_data, f, ensure_ascii=False, indent=2)  # 保存输入信息json文件
    f.close()
    s = (roi_tag[2] - roi_tag[0]) / 200 # 根据框子大小决定字号和线条粗细。
    cv2.rectangle(img_tag_, (int(roi_tag[0]), int(roi_tag[1])),
                    (int(roi_tag[2]), int(roi_tag[3])), (0, 0, 255), thickness=round(s*2))
    cv2.putText(img_tag_, "roi", (int(roi_tag[0]), int(roi_tag[1]-s)),
                    cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 255), thickness=round(s))
    map_o = config_object_name.OBJECT_MAP
    for bbox in bboxes:
        coor = bbox["coor"]; label = bbox["label"]
        s = int((coor[2] - coor[0]) / 3) # 根据框子大小决定字号和线条粗细。
        cv2.rectangle(img_tag_, (int(coor[0]), int(coor[1])),
                    (int(coor[2]), int(coor[3])), color_dict[label], thickness=round(s/50))
        # cv2.putText(img, label, (int(coor[0])-5, int(coor[1])-5),
        img_tag_ = img_chinese(img_tag_, map_o[input_data["type"]][label], (coor[0], coor[1]-s), color=color_dict[label], size=s)
    cv2.imwrite(os.path.join(save_path, "img_tag_cfg.jpg"), img_tag_)

    ## 输出可视化结果的图片。
    out_data["img_result"] = img2base64(img_tag_)

    return out_data


if __name__ == '__main__':
    tag_file = "/home/yh/inspection/python_codes/inspection_result/led/09-29-20-01-59/img_ref.jpg"
    ref_file = "test/p2.jpg"
    img_tag = img2base64(cv2.imread(tag_file))
    img_ = cv2.imread(ref_file)
    img_ref = img2base64(img_)
    ROI = [1249, 1154, 1885, 1400]
    W = img_.shape[1]; H = img_.shape[0]
    roi = [ROI[0]/W, ROI[1]/H, ROI[2]/W, ROI[3]/H]

    input_data = {"image": img_tag, "config":{}, "type": "led"} # "img_ref": img_ref, "bboxes": {"roi": roi}
    out_data = inspection_object_detection(input_data)
    print("inspection_object_detection result:")
    print("-----------------------------------------------")
    for s in out_data:
        if s != "img_result":
            print(s,":",out_data[s])
    print("----------------------------------------------")
    



