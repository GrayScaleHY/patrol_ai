import os
import time
import cv2
import json
from lib_image_ops import base642img, img2base64, img_chinese
from lib_inference_yolov5 import load_yolov5_model, inference_yolov5
from app_inspection_disconnector import sift_match, convert_coor
import numpy as np
import config_object_name
from lib_help_base import color_list

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
    if "img_ref" in input_data["config"]:
        img_ref = base642img(input_data["config"]["img_ref"])
    else:
        img_ref = None

    ## 感兴趣区域
    if "bboxes" in input_data["config"]:
        W = img_ref.shape[1]; H = img_ref.shape[0]
        roi = input_data["config"]["bboxes"]["roi"]
        roi = [int(roi[0]*W), int(roi[1]*H), int(roi[2]*W), int(roi[3]*H),]
    else:
        roi = None
    
    return img_tag, img_ref, roi

yolov5_meter = load_yolov5_model("/data/inspection/yolov5/meter.pt") # 加载仪表yolov5模型
yolov5_counter= load_yolov5_model("/data/inspection/yolov5/counter.pt") # 加载记数yolov5模型
yolov5_digital= load_yolov5_model("/data/inspection/yolov5/digital.pt") # 加载led数字yolov5模型

def inspection_counter(input_data):
    """
    动作次数数字识别。
    """
    ## 初始化输入输出信息。
    img_tag, img_ref, roi = get_input_data(input_data)
    out_data = {"code": 0, "data":{}, "msg": "Success request counter"} # 初始化out_data

    if input_data["type"] == "counter":
        yolov5_model = yolov5_counter
    elif input_data["type"] == "digital":
        yolov5_model = yolov5_digital
    else:
        out_data["msg"] = out_data["msg"] + "Type isn't counter; "
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
        if input_data["type"] == "digital":
            roi_tag = [0,0, img_tag.shape[1], img_tag.shape[0]]
        else:
            bbox_meters = inference_yolov5(yolov5_meter, img_tag, resize=640)
            if len(bbox_meters) == 0:
                roi_tag = [0,0, img_tag.shape[1], img_tag.shape[0]]
            else:
                roi_tag = bbox_meters[0]["coor"]
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
        out_data["msg"] = out_data["msg"] + "; Not find counter"
        return out_data
    
    ## 将bboxes映射到原图坐标
    bboxes = []
    for bbox in boxes:
        c = bbox["coor"]; r = roi_tag
        coor = [c[0]+r[0], c[1]+r[1], c[2]+r[0], c[3]+r[1]]
        bboxes.append({"label": bbox["label"], "coor": coor, "score": bbox["score"]})

    ## 根据从左到右的规则对bbox_digitals的存放排序
    l = [a['coor'][0] for a in bboxes]
    rank = [index for index,value in sorted(list(enumerate(l)),key=lambda x:x[1])]

    ## 将vals和bboxes添加进out_data
    vals = [boxes[i]['label'] for i in rank]
    bboxes_cfg = [boxes[i]['coor'] for i in rank]

    out_data['data'] = {"type": "counter", "values": vals, "bboxes": bboxes_cfg}

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
        img_tag_ = img_chinese(img_tag_, map_o[input_data["type"]][label], (coor[0], coor[1]-s*2), color=color_dict[label], size=s*2)
    cv2.imwrite(os.path.join(save_path, "img_tag_cfg.jpg"), img_tag_)

    ## 输出可视化结果的图片。
    out_data["img_result"] = img2base64(img_tag)

    return out_data

if __name__ == '__main__':
    
    tag_file = "test/2021_4_11_meter_pachong_1689_0_meter.jpg"
    ref_file = "test/#0182_org.jpg"
    img_tag = img2base64(cv2.imread(tag_file))
    img_ = cv2.imread(ref_file)
    img_ref = img2base64(img_)
    ROI = [908, 409, 1103, 484]
    W = img_.shape[1]; H = img_.shape[0]
    roi = [ROI[0]/W, ROI[1]/H, ROI[2]/W, ROI[3]/H]

    input_data = {"image": img_tag, "config":{}, "type": "counter"} # "img_ref": img_ref, "bboxes": {"roi": roi}
    out_data = inspection_counter(input_data)
    print("inspection_counter result:")
    print("-----------------------------------------------")
    for s in out_data:
        if s != "img_result":
            print(s,":",out_data[s])
    print("----------------------------------------------")