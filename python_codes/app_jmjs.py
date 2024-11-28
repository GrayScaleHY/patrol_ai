"""
sb_bx(设备变形), sb_dl(设备断裂), sb_qx(设备倾斜)
rq_ry(越线\闯入), js_dm(积水), rq_yw(异物入侵), hzyw(场地烟火), rq_xdw(小动物)
"""

import cv2
import json
from lib_image_ops import img_chinese
from lib_inference_yolov8 import load_yolov8_model, inference_yolov8
from lib_help_base import GetInputData, creat_img_result
from config_object_name import jmjs_dict
import numpy as np

yolov8_rec_defect = load_yolov8_model("/data/PatrolAi/yolov8/rec_defect.pt") # 缺陷
yolov8_coco = load_yolov8_model("/data/PatrolAi/yolov8/coco.pt") # coco模型
yolov8_xdw = load_yolov8_model("/data/PatrolAi/yolov8/xdw_js.pt") # 小动物 积水模型

rec_list = ["sb_bx", "sb_dl", "sb_qx", "hzyw", "rq_yw", "wcgz", "wcaqm", "sly_bjbmyw", "sly_dmyw"]
xdw_list = ["js_dm", "rq_xdw"]
coco_list = ["rq_ry", "rq_xdw", "ryjjph"]

def patrolai_jmjs(input_data):
    """
    yolov8的目标检测推理。
    """
    ## 提取输入请求信息
    DATA = GetInputData(input_data)
    an_type = DATA.type
    checkpoint = DATA.checkpoint
    img_tag = DATA.img_tag
    label_list = DATA.label_list

    ## 初始化out_data
    out_data = {"code": 0, "data":{"no_roi": [{}]}, "img_result": input_data["image"], "msg": "Request; "}

    ## 画上点位名称
    img_tag_ = img_tag.copy()
    img_tag_ = img_chinese(img_tag_, an_type + "_" + checkpoint , (10, 100), color=(255, 0, 0), size=30)

    yolov8_models = []
    if len(list(set(label_list) & set(rec_list))) > 0:
        yolov8_models.append(yolov8_rec_defect)
    if len(list(set(label_list) & set(xdw_list))) > 0:
        yolov8_models.append(yolov8_xdw)
    if len(list(set(label_list) & set(coco_list))) > 0:
        yolov8_models.append(yolov8_coco)
    
    labels = []
    for name in label_list:
        if name in jmjs_dict:
            labels = labels + jmjs_dict[name]["labels"]
    
    if len(yolov8_models) == 0:
        out_data["msg"] = out_data["msg"] + "labels isn't match; "
        out_data["code"] = 1
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 130), color=(255, 0, 0), size=30)
        out_data["img_result"] = creat_img_result(input_data, img_tag_) # 返回结果图
        return out_data
    
    cfgs = []
    for yolov8_model in yolov8_models:
        cfg = inference_yolov8(yolov8_model, img_tag, resize=640, focus_labels=labels, conf_thres=0.2) # inference
        cfgs = cfgs + cfg
    
    out_cfgs = []
    for cfg in cfgs: 
        label_en = cfg["label"]
        bbox = cfg["coor"]
        score = cfg["score"]
        for name in jmjs_dict:
            if label_en in jmjs_dict[name]["labels"]:
                label_en = name
                label_cn = jmjs_dict[name]["name"]
                out_cfg = {"label": label_cn, "label_en": label_en, "bbox": bbox, "score": score}
                out_cfgs.append(out_cfg)
                break
    
    if "ryjjph" in label_list:
        box_list = [cfg["coor"] for cfg in cfgs if cfg["label"] == "person"]
        b = np.array(box_list)
        box = [int(min(b[:,0])),int(min(b[:,1])),int(max(b[:,2])),int(max(b[:,3]))]
        if len(box_list) > 1:
            out_cfgs = [{"label": "人员聚集徘徊", "label_en": "ryjjph", "bbox": box, "score": 1.0}]
        else:
            out_cfgs = []

    
    out_data["data"]["no_roi"] = out_cfgs

    if len(out_cfgs) > 0:
        out_data["code"] = 1

    for cfg in out_cfgs:
        label_cn = cfg["label"]
        c = [int(i) for i in cfg["bbox"]]
        cv2.rectangle(img_tag_, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), (0,0,255), thickness=2)
        s = int((c[2] - c[0]) / 6) # 根据框子大小决定字号和线条粗细。
        img_tag_ = img_chinese(img_tag_, label_cn, (c[0], c[1]), color=(0,0,255), size=s)
    
    img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 130), color=(255, 0, 0), size=30)

    out_data["img_result"] = creat_img_result(input_data, img_tag_) # 返回结果图
    
    return out_data

if __name__ == '__main__':
    from lib_help_base import get_save_head, save_input_data, save_output_data
    json_file = "/data/PatrolAi/result_patrol/0330051344_鸟巢点位测试_input_data.json"
    # f = open(json_file,"r",encoding='utf-8')
    # input_data = json.load(f)
    # f.close()
    input_data = {
    "checkpoint": "设备缺陷识别",
    "image": "/data/PatrolAi/test_images/ryjj.jpg",
    "config": {
        "label_list": [
            "ryjjph"
        ]
    },
    "type": "rec_defect"
}
    
    out_data = patrolai_jmjs(input_data)
    save_dir, name_head = get_save_head(input_data)
    save_input_data(input_data, save_dir, name_head, draw_img=True)
    save_output_data(out_data, save_dir, name_head)
    print("-----------------------------------------------")
    for s in out_data:
        if s != "img_result":
            print(s,":",out_data[s])
    print("----------------------------------------------")
