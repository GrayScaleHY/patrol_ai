import cv2
import json
from lib_image_ops import  img_chinese
from lib_rcnn_ops import check_iou
from lib_inference_yolov8 import load_yolov8_model, inference_yolov8
from lib_img_registration import roi_registration
import config_object_name
from lib_help_base import GetInputData, is_include, color_list, creat_img_result, draw_region_result, reg_crop
from lib_model_import import model_load
from config_model_list import model_threshold_dict
## 开关柜标签状态检测


def inspection_sticker_detection(input_data):
    """
    yolov8的目标检测推理。
    """
    ## 提取输入请求信息
    DATA = GetInputData(input_data)
    checkpoint = DATA.checkpoint;
    an_type = DATA.type
    img_tag = DATA.img_tag;
    img_ref = DATA.img_ref
    roi = DATA.roi;
    reg_box = DATA.regbox

    ## 初始化out_data
    out_data = {"code": 0, "data": {}, "img_result": input_data["image"], "msg": "Request; "}

    ## 画上点位名称
    img_tag_ = img_tag.copy()
    img_tag_ = img_chinese(img_tag_, an_type + "_" + checkpoint, (10, 100), color=(255, 0, 0), size=30)

    if an_type == "sticker":
        yolov8_model=model_load(an_type)
        labels_dict = yolov8_model.names
        conf = model_threshold_dict[an_type]
        labels = [labels_dict[id] for id in labels_dict]
        #model_type = "sticker"
    else:
        out_data["msg"] = out_data["msg"] + "Type isn't object; "
        out_data["code"] = 1
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 130), color=(255, 0, 0), size=30)
        out_data["img_result"] = creat_img_result(input_data, img_tag_)  # 返回结果图
        return out_data

    # img_ref截取regbox区域用于特征匹配
    if reg_box and len(reg_box) != 0:
        img_ref = reg_crop(img_ref, *reg_box)

    ## 求出目标图像的感兴趣区域
    roi_tag, _ = roi_registration(img_ref, img_tag, roi)
    for name, c in roi_tag.items():
        cv2.rectangle(img_tag_, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (255, 0, 255), thickness=1)
        # cv2.putText(img_tag_, name, (int(c[0]), int(c[1])+10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)
        s = int((c[2] - c[0]) / 10)  # 根据框子大小决定字号和线条粗细。
        img_tag_ = img_chinese(img_tag_, name, (c[0], c[1]), color=(255, 0, 255), size=s)

    ## 模型推理
    cfgs = inference_yolov8(yolov8_model, img_tag, focus_labels=labels, conf_thres=conf)  # inference
    cfgs = check_iou(cfgs, iou_limit=0.5)  # 增加iou机制


    #未识别到标签，data设False
    if len(cfgs) == 0:
        out_data["code"] = 0
        for name ,c in roi_tag.items():
            out_data['data'][name]=[
                {'label':"False","bbox":[int(c[0]), int(c[1]), int(c[2]), int(c[3])]}
            ]
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 130), color=(255, 0, 0), size=30)
        out_data["img_result"] = creat_img_result(input_data, img_tag_)  # 返回结果图
        return out_data

    ## labels 列表 和 color 列表
    color_dict = {}
    # name_dict = {}
    for i, label in enumerate(labels):
        color_dict[label] = color_list(len(labels))[i]
    #     if label in config_object_name.OBJECT_MAP[model_type]:
    #         name_dict[label] = config_object_name.OBJECT_MAP[model_type][label]
    #     else:
    #         name_dict[label] = label

    ## 判断bbox是否在roi中
    for name, roi in roi_tag.items():
        out_data["data"][name] = []
        for cfg in cfgs:
            if is_include(cfg["coor"], roi, srate=0.8):
                c = cfg["coor"]
                label = cfg["label"]
                # cfg_out = name_dict[label]
                out_data["data"][name].append(label)

                # 画出识别框
                cv2.rectangle(img_tag_, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), color_dict[label], thickness=2)
                s = int((c[2] - c[0]) / 6)  # 根据框子大小决定字号和线条粗细。
                img_tag_ = img_chinese(img_tag_, label, (c[0], c[1]), color=color_dict[label], size=s)
        #判断out_data["data"][name]中是否同时有两个表标签，返回label对应设为True/False
        out_data["data"][name]=list(set(out_data["data"][name]))
        if len(out_data["data"][name])==2:
            out_data["data"][name]=[
                {'label':"True","bbox":[int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])]}
            ]
        else:
            out_data["data"][name] = [
                {'label': "False", "bbox": [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])]}
            ]



    ## 主从逻辑中，每个roi框都画一张图
    out_data = draw_region_result(out_data, input_data, roi_tag)

    ## 老版本的接口输出，"data"由字典改为list
    no_roi = [name.startswith("old_roi") for name in out_data["data"]]
    if all(no_roi):  ## 全为1， 返回True
        _cfgs = []
        for name, _cfg in out_data["data"].items():
            if len(_cfg) > 0:
                _cfgs.append(_cfg[0])
        out_data["data"] = _cfgs
        if out_data["data"] == [{}]:
            out_data["data"] = []

    img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 130), color=(255, 0, 0), size=30)

    out_data["img_result"] = creat_img_result(input_data, img_tag_)  # 返回结果图

    return out_data


if __name__ == '__main__':
    from lib_help_base import get_save_head, save_input_data, save_output_data

    json_file = "/data/PatrolAi/result_patrol/0330051344_鸟巢点位测试_input_data.json"
    f = open(json_file,"r",encoding='utf-8')
    input_data = json.load(f)
    f.close()


    out_data = inspection_sticker_detection(input_data)
    save_dir, name_head = get_save_head(input_data)
    save_input_data(input_data, save_dir, name_head, draw_img=True)
    save_output_data(out_data, save_dir, name_head)
    print("-----------------------------------------------")
    for s in out_data:
        if s != "img_result":
            print(s, ":", out_data[s])
    print("----------------------------------------------")




