import os
import time
import cv2
import json
from lib_image_ops import base642img, img2base64, img_chinese
from lib_inference_yolov5 import load_yolov5_model, inference_yolov5, check_iou
from lib_help_base import color_list
from lib_sift_match import sift_match, convert_coor, sift_create
import config_object_name
from config_object_name import convert_label
import numpy as np
## 表计， 二次设备，17类缺陷, 安全帽， 烟火
from config_load_models_var import yolov5_meter, \
                                   yolov5_ErCiSheBei, \
                                   yolov5_rec_defect_x6, \
                                   yolov5_coco, \
                                   yolov5_ShuZiBiaoJi, \
                                   yolov5_dztx
                                #    yolov5_jmjs, \
                                #    yolov5_helmet, \
                                #    yolov5_fire_smoke, \
                                #    yolov5_led_color

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

def rank_digital(obj_data, obj_type="counter"):
    """
    args:
        obj_data: 常规目标检测输出格式，[{"label": "0", "bbox": [xmin,ymin,xmax,ymax], "score":0.635}, ..]
        obj_type: counter or digital
    return:
        new_data: 数字类排好序的格式，{"type": "counter", "values": ['6', '5'], "bboxes": [[xmin,ymin,xmax,ymax], ..]}
    """
    l = [cfg["bbox"][0] for cfg in obj_data]
    rank = [index for index,value in sorted(list(enumerate(l)),key=lambda x:x[1])]
    vals = [obj_data[i]["label"] for i in rank]
    bboxes = [obj_data[i]["bbox"] for i in rank]
    new_data = {"type": obj_type, "values": vals, "bboxes": bboxes}
    return new_data
    
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
            if len(input_data["config"]["img_ref"]) > 10:
                img_ref = base642img(input_data["config"]["img_ref"])
        
    ## 感兴趣区域
    roi = None # 初始假设
    if "bboxes" in input_data["config"]:
        if isinstance(input_data["config"]["bboxes"], dict):
            if "roi" in input_data["config"]["bboxes"]:
                if isinstance(input_data["config"]["bboxes"]["roi"], list):
                    if isinstance(input_data["config"]["bboxes"]["roi"][0], list):
                        roi = input_data["config"]["bboxes"]["roi"][0]
                    else:
                        roi = input_data["config"]["bboxes"]["roi"]
                    W = img_ref.shape[1]; H = img_ref.shape[0]
                    roi = [int(roi[0]*W), int(roi[1]*H), int(roi[2]*W), int(roi[3]*H)]  
    
    ## 设备状态与显示名字的映射关系。
    status_map = None
    if "status_map" in input_data["config"]:
        if isinstance(input_data["config"]["status_map"], dict):
            status_map = input_data["config"]["status_map"]

    ## 指定label_list。
    label_list = None
    if "label_list" in input_data["config"]:
        if isinstance(input_data["config"]["label_list"], list):
            label_list = input_data["config"]["label_list"]
    
    return img_tag, img_ref, roi, status_map, label_list

def inspection_object_detection(input_data):
    """
    yolov5的目标检测推理。
    """
    ## 将输入请求信息可视化
    TIME_START = time.strftime("%m%d%H%M%S") + "_"
    if "checkpoint" in input_data and isinstance(input_data["checkpoint"], str) and len(input_data["checkpoint"]) > 0:
        TIME_START = TIME_START + input_data["checkpoint"] + "_"
    save_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_path = os.path.join(save_path, "result_patrol", input_data["type"])
    os.makedirs(save_path, exist_ok=True)
    f = open(os.path.join(save_path, TIME_START + "input_data.json"), "w")
    json.dump(input_data, f, ensure_ascii=False)  # 保存输入信息json文件
    f.close()

    ## 初始化输入输出信息。
    img_tag, img_ref, roi, status_map, label_list = get_input_data(input_data)
    out_data = {"code": 0, "data":[], "img_result": input_data["image"], "msg": "Success request object detect; "} # 初始化out_data
    if input_data["type"] == "rec_defect" or input_data["type"] == "fire_smoke":
        out_data = {"code": 1, "data":[], "img_result": input_data["image"], "msg": "Success request object detect; "} # 初始化out_data

    ## 将输入请求信息可视化
    img_tag_ = img_tag.copy()
    img_tag_ = img_chinese(img_tag_, TIME_START + input_data["type"] , (10, 10), color=(255, 0, 0), size=60)
    cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag.jpg"), img_tag) # 将输入图片可视化
    if img_ref is not None:
        cv2.imwrite(os.path.join(save_path, TIME_START + "img_ref.jpg"), img_ref) # 将输入图片可视化
    if roi is not None:   # 如果配置了感兴趣区域，则画出感兴趣区域
        img_ref_ = img_ref.copy()
        cv2.rectangle(img_ref_, (int(roi[0]), int(roi[1])),(int(roi[2]), int(roi[3])), (255, 0, 255), thickness=1)
        cv2.putText(img_ref_, "roi", (int(roi[0]), int(roi[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)
        cv2.imwrite(os.path.join(save_path, TIME_START + "img_ref_cfg.jpg"), img_ref_)

    ## 选择模型
    if input_data["type"] == "meter":
        yolov5_model = yolov5_meter
        labels_dict = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
        labels = [labels_dict[id] for id in labels_dict]
        model_type = "meter"
    # elif input_data["type"] == "fire_smoke":
    #     yolov5_model = yolov5_fire_smoke
    #     labels_dict = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
    #     labels = [labels_dict[id] for id in labels_dict]
    #     model_type = "fire_smoke"
    # elif input_data["type"] == "helmet":
    #     yolov5_model = yolov5_helmet
    #     labels_dict = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
    #     labels = [labels_dict[id] for id in labels_dict]
    #     model_type = "helmet"
    # elif input_data["type"] == "led_color":
    #     yolov5_model = yolov5_led_color
    #     labels_dict = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
    #     labels = [labels_dict[id] for id in labels_dict]
    #     model_type = "led"
    elif input_data["type"] == "digital":
        yolov5_model = yolov5_ShuZiBiaoJi
        labels_dict = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
        labels = [labels_dict[id] for id in labels_dict]
        model_type = "digital"
    elif input_data["type"] == "counter":
        yolov5_model = yolov5_ShuZiBiaoJi
        labels = ["0","1","2","3","4","5","6","7","8","9"]
        model_type = "counter"
    elif input_data["type"] == "rec_defect":
        if label_list == ["xdwcr"]:
            yolov5_model = yolov5_coco
            labels = ["bird", "cat", "dog", "sheep"]
            model_type = "meter"
        # elif label_list == ["hzyw"]:
        #     yolov5_model = yolov5_fire_smoke
        #     labels_dict = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
        #     labels = [labels_dict[id] for id in labels_dict]
        #     model_type = "fire_smoke"
        # elif label_list == ["sb_bx"] or label_list == ["sb_dl"] or label_list == ["sb_qx"]:
        #     yolov5_model = yolov5_jmjs
        #     labels = label_list
        #     model_type = "jmjs"
        else:
            yolov5_model = yolov5_rec_defect_x6
            labels_dict = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
            labels = [labels_dict[id] for id in labels_dict]
            if label_list is not None:
                labels = [convert_label(l, "rec_defect") for l in label_list]
            model_type = "rec_defect"

    elif input_data["type"] == "pressplate": 
        yolov5_model = yolov5_ErCiSheBei
        labels = ["kgg_ybh", "kgg_ybf"]
        model_type = "ErCiSheBei"
    elif input_data["type"] == "air_switch":
        yolov5_model = yolov5_ErCiSheBei
        labels = ["kqkg_hz", "kqkg_fz"]
        model_type = "ErCiSheBei"
    elif input_data["type"] == "led":
        yolov5_model = yolov5_ErCiSheBei
        labels = ["zsd_l", "zsd_m"]
        model_type = "ErCiSheBei"
    elif input_data["type"] == "fanpaiqi":
        yolov5_model = yolov5_ErCiSheBei
        model_type = "ErCiSheBei"
        labels = ["fpq_h", "fpq_f", "fpq_jd"]
    elif input_data["type"] == "rotary_switch":
        yolov5_model = yolov5_ErCiSheBei
        labels = ["xnkg_s", "xnkg_zs", "xnkg_ys", "xnkg_z"]
        model_type = "ErCiSheBei"
    elif input_data["type"] == "door":
        yolov5_model = yolov5_ErCiSheBei
        labels = ["xmbhyc", "xmbhzc"]
        model_type = "ErCiSheBei"
    elif input_data["type"] == "key":
        yolov5_model = yolov5_ErCiSheBei
        labels = ["ys"]
        model_type = "ErCiSheBei"
    elif input_data["type"] == "disconnector_notemp":
        yolov5_model = yolov5_dztx
        labels = ["he","fen","budaowei"]
        model_type = "disconnector_texie"
    elif input_data["type"] == "person":
        yolov5_model = yolov5_coco
        labels = ["person"]
        model_type = "coco"
    else:
        out_data["msg"] = out_data["msg"] + "Type isn't object; "
        out_data["code"] = 1
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
        out_data["img_result"] = img2base64(img_tag_)
        cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag_cfg.jpg"), img_tag_)
        return out_data

    if "augm" in input_data["config"]:
        if isinstance(input_data["config"]["augm"], list):
            if len(input_data["config"]["augm"]) == 2:
                augm = input_data["config"]["augm"]
                augm = [float(augm[0]), float(augm[1])]
                img_tag = np.uint8(np.clip((augm[0] * img_tag + augm[1]), 0, 255))

    ## 生成目标检测信息
    if input_data["type"] == "rec_defect":
        if label_list == ["hzyw"] or label_list == ["xdwcr"]:
            cfgs = inference_yolov5(yolov5_model, img_tag, resize=640, pre_labels=labels) # inference
        else:
            cfgs = inference_yolov5(yolov5_model, img_tag, resize=1280, pre_labels=labels, conf_thres=0.7) # inference
    else:
        cfgs = inference_yolov5(yolov5_model, img_tag, resize=640, pre_labels=labels, conf_thres=0.3) # inference
    cfgs = check_iou(cfgs, iou_limit=0.5) # 增加iou机制

    if len(cfgs) == 0: #没有检测到目标
        out_data["msg"] = out_data["msg"] + "; Not find object"
        if input_data["type"] == "rec_defect" or input_data["type"] == "fire_smoke":
            out_data["code"] = 0
        else:
            out_data["code"] = 1
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
        out_data["img_result"] = img2base64(img_tag_)
        cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag_cfg.jpg"), img_tag_)
        return out_data

    ## labels 列表 和 color 列表
    colors = color_list(len(labels))
    color_dict = {}
    name_dict = {}
    for i, label in enumerate(labels):
        color_dict[label] = colors[i]
        if status_map is not None and label in status_map:
            name_dict[label] = status_map[label]
        elif label in config_object_name.OBJECT_MAP[model_type]:
            name_dict[label] = config_object_name.OBJECT_MAP[model_type][label]
        else:
            name_dict[label] = label

        ## 如果有"real_val"，则输出real_val的值
        if "real_val" in input_data["config"] and isinstance(input_data["config"]["real_val"], str):
            name_dict[label] = input_data["config"]["real_val"]

    ## 画出boxes
    for cfg in cfgs:
        c = cfg["coor"]; label = cfg["label"]
        cv2.rectangle(img_tag_, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), color_dict[label], thickness=2)
        
        if input_data["type"] == "counter" or input_data["type"] == "digital":
            s = int(c[2] - c[0]) # 根据框子大小决定字号和线条粗细。
            img_tag_ = img_chinese(img_tag_, name_dict[label], (c[0], c[1]-s), color=color_dict[label], size=s)
        else:
            s = int((c[2] - c[0]) / 6) # 根据框子大小决定字号和线条粗细。
            img_tag_ = img_chinese(img_tag_, name_dict[label], (c[0], c[1]), color=color_dict[label], size=s)

    ## 求出目标图像的感兴趣区域
    if roi is not None:
        feat_ref = sift_create(img_ref, rm_regs=[[0,0,1,0.1],[0,0.9,1,1]])
        feat_tag = sift_create(img_tag, rm_regs=[[0,0,1,0.1],[0,0.9,1,1]])
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

    if input_data["type"] == "counter" or input_data["type"] == "digital":
        out_data["data"] = rank_digital(out_data["data"], obj_type=input_data["type"])

    if input_data["type"] == "key":
        out_data["data"] = {"label": input_data["type"], "number": len(bboxes), "boxes": bboxes}
    
    ## 可视化计算结果
    f = open(os.path.join(save_path, TIME_START + "out_data.json"), "w")
    json.dump(out_data, f, ensure_ascii=False, indent=2)  # 保存输入信息json文件
    f.close()
    
    ## 输出可视化结果的图片。
    img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
    out_data["img_result"] = img2base64(img_tag_)
    cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag_cfg.jpg"), img_tag_)
    
    return out_data

if __name__ == '__main__':
    json_file = "/data/PatrolAi/result_patrol/12-15-11-30-07_input_data.json"
    f = open(json_file,"r",encoding='utf-8')
    input_data = json.load(f)
    f.close()
    out_data = inspection_object_detection(input_data)
    print("inspection_object_detection result:")
    print("-----------------------------------------------")
    for s in out_data:
        if s != "img_result":
            print(s,":",out_data[s])
    print("----------------------------------------------")
    



