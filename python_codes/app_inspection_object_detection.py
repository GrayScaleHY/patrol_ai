import os
import time
import cv2
import json
from lib_image_ops import base642img, img2base64, img_chinese
from lib_inference_yolov5 import load_yolov5_model, inference_yolov5
from lib_help_base import color_list
from lib_sift_match import sift_match, convert_coor, sift_create
import config_object_name
import numpy as np

def is_include(sub_box, par_box, srate=0.8):
    
    sb = sub_box; pb = par_box
    sb = [min(sb[0],sb[2]), min(sb[1],sb[3]), max(sb[0],sb[2]), max(sb[1],sb[3])]
    pb = [min(pb[0],pb[2]), min(pb[1],pb[3]), max(pb[0],pb[2]), max(pb[1],pb[3])]

    ## 至少一个点在par_box里面
    points = [[sb[0],sb[1]], [sb[2],sb[1]], [sb[0],sb[3]], [sb[2],sb[3]]]
    is_in = False
    for p in points:
        if p[0] >= pb[0] and p[0] <= pb[2] and p[1] >= pb[1] and p[1] <= pb[3]:
            is_in = True
    if not is_in:
        return False
    
    ## 判断交集占多少
    xmin = max(pb[0], sb[0]); ymin = max(pb[1], sb[1])
    xmax = min(pb[2], sb[2]); ymax = min(pb[3], sb[3])
    s_include = (xmax-xmin) * (ymax-ymin)
    s_box = (sb[2]-sb[0]) * (sb[3]-sb[1])
    if s_include / s_box >= srate:
        return True
    else:
        return False
    
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
    img_ref = None
    if "img_ref" in input_data["config"]:
        if isinstance(input_data["config"]["img_ref"], str):
            img_ref = base642img(input_data["config"]["img_ref"])      
        
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
    
    ## 设备状态与显示名字的映射关系。
    status_map = None
    if "status_map" in input_data["config"]:
        if isinstance(input_data["config"]["status_map"], dict):
            status_map = input_data["config"]["status_map"]
    
    ## 是否附真实值。
    real_val = None
    if "real_val" in input_data["config"]:
        if isinstance(input_data["config"]["real_val"], dict):
            real_val = input_data["config"]["real_val"]
    
    return img_tag, img_ref, roi, status_map, real_val

yolov5_meter = load_yolov5_model("/data/inspection/yolov5/meter.pt") # 表盘
yolov5_ErCiSheBei = load_yolov5_model("/data/inspection/yolov5/ErCiSheBei.pt") ## 二次设备状态模型
# yolov5_fire_smoke = load_yolov5_model("/data/inspection/yolov5/fire_smoke.pt") # 烟火
# yolov5_pressplate = load_yolov5_model("/data/inspection/yolov5/pressplate.pt") # 压板
# yolov5_helmet = load_yolov5_model("/data/inspection/yolov5/helmet.pt") # 安全帽
yolov5_rec_defect = load_yolov5_model("/data/inspection/yolov5/rec_defect_x6.pt") # 北京送检17类缺陷

def inspection_object_detection(input_data):
    """
    yolov5的目标检测推理。
    """
    ## 将输入请求信息可视化
    TIME_START = time.strftime("%m-%d-%H-%M-%S") 
    save_path = os.path.join("inspection_result", input_data["type"], TIME_START)
    os.makedirs(save_path, exist_ok=True)
    f = open(os.path.join(save_path, "input_data.json"), "w")
    json.dump(input_data, f, ensure_ascii=False)  # 保存输入信息json文件
    f.close()

    ## 初始化输入输出信息。
    img_tag, img_ref, roi, status_map, real_val = get_input_data(input_data)
    out_data = {"code": 0, "data":[], "img_result": input_data["image"], "msg": "Success request object detect; "} # 初始化out_data
    ## 将输入请求信息可视化
    img_tag_ = img_tag.copy()
    cv2.imwrite(os.path.join(save_path, "img_tag.jpg"), img_tag) # 将输入图片可视化
    if img_ref is not None:
        cv2.imwrite(os.path.join(save_path, "img_ref.jpg"), img_ref) # 将输入图片可视化
    if roi is not None:   # 如果配置了感兴趣区域，则画出感兴趣区域
        img_ref_ = img_ref.copy()
        cv2.rectangle(img_ref_, (int(roi[0]), int(roi[1])),(int(roi[2]), int(roi[3])), (255, 0, 255), thickness=1)
        cv2.putText(img_ref_, "roi", (int(roi[0]), int(roi[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)
        cv2.imwrite(os.path.join(save_path, "img_ref_cfg.jpg"), img_ref_)

    ## 选择模型
    if input_data["type"] == "meter":
        yolov5_model = yolov5_meter
        labels = ["meter"]
        model_type = "meter"
    elif input_data["type"] == "pressplate": 
        yolov5_model = yolov5_ErCiSheBei
        labels = ["kgg_ybh", "kgg_ybf"]
        model_type = "ErCiSheBei"
    elif input_data["type"] == "air_switch":
        yolov5_model = yolov5_ErCiSheBei
        labels = ["kqkg_hz", "kqkg_fz"]
        model_type = "ErCiSheBei"
    # elif input_data["type"] == "fire_smoke":
    #     yolov5_model = yolov5_fire_smoke
    elif input_data["type"] == "led":
        yolov5_model = yolov5_ErCiSheBei
        labels = ["zsd_lvdl", "zsd_lvdm", "zsd_hongdl", "zsd_hongdm", "zsd_baidl", "zsd_baidm", "zsd_huangdl", "zsd_huangdm", "zsd_heidm"]
        model_type = "ErCiSheBei"
    # elif input_data["type"] == "helmet":
    #     yolov5_model = yolov5_helmet
    elif input_data["type"] == "fanpaiqi":
        yolov5_model = yolov5_ErCiSheBei
        model_type = "ErCiSheBei"
        labels = ["fpq_h", "fpq_f", "fpq_jd"]
    elif input_data["type"] == "rotary_switch":
        yolov5_model = yolov5_ErCiSheBei
        labels = ["xnkg_s", "xnkg_zs", "xnkg_ys", "xnkg_z"]
        model_type = "ErCiSheBei"
    # elif input_data["type"] == "arrow":
    #     yolov5_model = yolov5_rotary_switch
    elif input_data["type"] == "door":
        yolov5_model = yolov5_ErCiSheBei
        labels = ["xmbhyc", "xmbhzc"]
        model_type = "ErCiSheBei"
    elif input_data["type"] == "key":
        yolov5_model = yolov5_ErCiSheBei
        labels = ["xmbhyc", "xmbhzc"]
        model_type = "ErCiSheBei"
    # elif input_data["type"] == "robot":
    #     yolov5_model = yolov5_robot
    elif input_data["type"] == "rec_defect":
        yolov5_model = yolov5_rec_defect
        labels = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
        model_type = "rec_defect"
    else:
        out_data["msg"] = out_data["msg"] + "Type isn't object; "
        return out_data

    ## 生成目标检测信息
    if input_data["type"] == "rec_defect":
        cfgs = inference_yolov5(yolov5_model, img_tag, resize=1280, pre_labels=labels) # inference
    else:
        cfgs = inference_yolov5(yolov5_model, img_tag, resize=640, pre_labels=labels) # inference
    if len(cfgs) == 0: #没有检测到目标
        out_data["msg"] = out_data["msg"] + "; Not find object"
        return out_data

    ## labels 列表 和 color 列表
    colors = color_list(len(labels))
    color_dict = {}
    name_dict = {}
    for i, label in enumerate(labels):
        color_dict[label] = colors[i]
        if status_map is not None and label in status_map:
            name_dict[label] = status_map[label]
        else:
            name_dict[label] = config_object_name.OBJECT_MAP[model_type][label]

        ## 如果有"real_val"，则输出real_val的值
        if "real_val" in input_data["config"]:
            name_dict[label] = input_data["config"]["real_val"]

    ## 画出boxes
    for cfg in cfgs:
        c = cfg["coor"]; label = cfg["label"]
        s = int((c[2] - c[0]) / 4) # 根据框子大小决定字号和线条粗细。
        cv2.rectangle(img_tag_, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), color_dict[label], thickness=2)
        # cv2.putText(img, label, (int(coor[0])-5, int(coor[1])-5),
        img_tag_ = img_chinese(img_tag_, name_dict[label], (c[0], c[1]-s), color=color_dict[label], size=s)

    ## 求出目标图像的感兴趣区域
    if roi is not None:
        feat_ref = sift_create(img_ref)
        feat_tag = sift_create(img_tag)
        M = sift_match(feat_ref, feat_tag, ratio=0.5, ops="Perspective")
        if M is None:
            out_data["msg"] = out_data["msg"] + "; Not enough matches are found"
            roi_tag = roi
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

        ## 画出roi_tag
        c = roi_tag
        cv2.rectangle(img_tag_, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), (255,0,255), thickness=1)
        cv2.putText(img_tag_, "roi", (int(c[0]), int(c[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)

    ## 判断bbox是否在roi中
    bboxes = []
    for cfg in cfgs:
        if roi is None or is_include(cfg["coor"], roi_tag, srate=0.5):
            cfg_out = {"label": name_dict[cfg["label"]], "bbox": cfg["coor"], "score": float(cfg["score"])}
            out_data["data"].append(cfg_out)
            bboxes.append(cfg["coor"])

    if input_data["type"] == "key":
        out_data["data"] = {"label": input_data["type"], "number": len(bboxes), "boxes": bboxes}
    
    ## 可视化计算结果
    f = open(os.path.join(save_path, "out_data.json"), "w")
    json.dump(out_data, f, ensure_ascii=False, indent=2)  # 保存输入信息json文件
    f.close()
    cv2.imwrite(os.path.join(save_path, "img_tag_cfg.jpg"), img_tag_)

    ## 输出可视化结果的图片。
    out_data["img_result"] = img2base64(img_tag_)
    
    return out_data

if __name__ == '__main__':
    # tag_file = "/home/yh/inspection/python_codes/inspection_result/led/09-29-20-01-59/img_ref.jpg"
    # ref_file = "test/p2.jpg"
    # img_tag = img2base64(cv2.imread(tag_file))
    # img_ = cv2.imread(ref_file)
    # img_ref = img2base64(img_)
    # ROI = [1249, 1154, 1885, 1400]
    # W = img_.shape[1]; H = img_.shape[0]
    # roi = [ROI[0]/W, ROI[1]/H, ROI[2]/W, ROI[3]/H]

    # input_data = {"image": img_tag, "config":{}, "type": "led"} # "img_ref": img_ref, "bboxes": {"roi": roi}
    f = open("/home/yh/image/python_codes/test/disconnetor_test/pressplate/05-18-15-55-40/input_data.json", "r", encoding='utf-8')
    input_data = json.load(f)
    f.close()
    out_data = inspection_object_detection(input_data)
    print("inspection_object_detection result:")
    print("-----------------------------------------------")
    for s in out_data:
        if s != "img_result":
            print(s,":",out_data[s])
    print("----------------------------------------------")
    



