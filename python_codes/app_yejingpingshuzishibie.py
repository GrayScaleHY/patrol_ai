import os
import cv2
import time
import json
import math
from lib_image_ops import base642img, img2base64, img_chinese
from lib_help_base import oil_high
import numpy as np

from lib_inference_yolov5 import load_yolov5_model, inference_yolov5,check_iou
from lib_img_registration import registration, convert_coor
from lib_help_base import GetInputData

yolov5_mubiaokuang = load_yolov5_model("/data/PatrolAi/yolov5/shuzi_crop.pt")  # 数字表记寻框模型
yolov5_shuzishibie = load_yolov5_model("/data/PatrolAi/yolov5/shuzi_rec.pt")  # 数字表记数字识别模型
yolov5_jishukuang = load_yolov5_model("/data/PatrolAi/yolov5/jishu_crop.pt")  # 计数表寻框模型
yolov5_jishushibie = load_yolov5_model("/data/PatrolAi/yolov5/jishu_rec.pt")  # 计数表数字识别模型

def inspection_digital_rec(input_data):
    ## 初始化输入输出信息。
    TIME_START = time.strftime("%m%d%H%M%S") + "_"
    if "checkpoint" in input_data and isinstance(input_data["checkpoint"], str) and len(input_data["checkpoint"]) > 0:
        TIME_START = TIME_START + input_data["checkpoint"] + "_"
    # save_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # save_path = os.path.join(save_path, "result_patrol", input_data["type"])
    # os.makedirs(save_path, exist_ok=True)
    # with open(os.path.join(save_path, TIME_START + "input_data.json"), "w") as f:
    #     json.dump(input_data, f, ensure_ascii=False)  # 保存输入信息json文件

    out_data = {"code": 0, "data": {}, "msg": "Success"}  # 初始化输出信息

    ## 提取输入请求信息
    input_msg=GetInputData(input_data)
    roi=input_msg.roi
    img_ref=input_msg.img_ref
    img_tag=input_msg.img_tag
    # img_tag, img_ref, roi, cc, bc = get_input_data(input_data)

    ## 将输入请求信息可视化
    img_tag_ = img_tag.copy()
    img_tag_ = img_chinese(img_tag_, TIME_START + input_data["type"], (10, 10), color=(255, 0, 0), size=60)

    if input_data["type"] != "digital" and input_data["type"] != "counter":
        out_data["msg"] = out_data["msg"] + "type isn't digital or counter; "
        out_data["code"] = 1
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
        out_data["img_result"] = img2base64(img_tag_)
        # cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag_cfg.jpg"), img_tag_)
        return out_data

    yolo_crop,yolo_rec=yolov5_mubiaokuang,yolov5_shuzishibie
    if input_data["type"] != "digital":
        yolo_crop,yolo_rec=yolov5_jishukuang,yolov5_jishushibie

    # cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag.jpg"), img_tag)

    if len(roi)!=0 and img_ref is not None:  ## 如果配置了感兴趣区域，则画出感兴趣区域
        img_ref_ = img_ref.copy()
        # cv2.imwrite(os.path.join(save_path, TIME_START + "img_ref.jpg"), img_ref_)
        for roi_index in roi:
            cv2.rectangle(img_ref_, (int(roi_index[0]), int(roi_index[1])), (int(roi_index[2]), int(roi_index[3])), (255, 0, 255), thickness=1)
            cv2.putText(img_ref_, "roi", (int(roi_index[0]), int(roi_index[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255),
                    thickness=1)
        # cv2.imwrite(os.path.join(save_path, TIME_START + "img_ref_cfg.jpg"), img_ref_)

        ## 如果没有配置roi，则自动识别表盘作为roi
    if len(roi)==0:
        M = None
    else:
        # 求偏移矩阵
        M = registration(img_ref, img_tag)
    roi_tag_list = []
    if M is None:
        roi_tag = [0, 0, img_tag.shape[1], img_tag.shape[0]]
    else:
        for roi_index in roi:
            coors = [(roi_index[0], roi_index[1]), (roi_index[2], roi_index[1]), (roi_index[2], roi_index[3]),
                     (roi_index[0], roi_index[3])]
            coors_ = []
            for coor in coors:
                coors_.append(list(convert_coor(coor, M)))
            xs = [coor[0] for coor in coors_]
            ys = [coor[1] for coor in coors_]
            xmin = max(0, min(xs))
            ymin = max(0, min(ys))
            xmax = min(img_tag.shape[1], max(xs))
            ymax = min(img_tag.shape[0], max(ys))
            roi_tag = [xmin, ymin, xmax, ymax]
            roi_tag_list.append(roi_tag)
    # img_roi = img_tag[int(roi_tag[1]): int(roi_tag[3]), int(roi_tag[0]): int(roi_tag[2])]

        ## 使用映射变换矫正目标图，并且转换坐标点。

        ## 将矫正偏移的信息写到图片中
    for roi_tag in roi_tag_list:
        cv2.rectangle(img_tag_, (int(roi_tag[0]), int(roi_tag[1])), (int(roi_tag[2]), int(roi_tag[3])), (255, 0, 255),
                        thickness=1)
        cv2.putText(img_tag_, "roi", (int(roi_tag[0]), int(roi_tag[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                    thickness=1)
        # print(img_roi.shape)
    value_list = []  # 输出数字列表
    bboxes_list = []  # 位置列表

    # 第一阶段区域识别，截取图像
    bbox_cfg = inference_yolov5(yolo_crop, img_tag_)
    # 未检测到目标
    if len(bbox_cfg) < 1:
        out_data["msg"] = out_data["msg"] + "Can not find digital; "
        out_data["code"] = 1
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
        out_data["img_result"] = img2base64(img_tag_)
        # cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag_cfg.jpg"), img_tag_)
        return out_data

        # 检测出的位置按y坐标排序，做640*640填充，二次识别
    coor_list = [item['coor'] for item in bbox_cfg]
    bboxes_list_sort = sorted(coor_list, key=lambda x: x[-1], reverse=False)
    # print("bboxes_list:",bboxes_list)

    for coor in bboxes_list_sort:
        #去除roi框外，不做识别
        x_middle = (coor[0] + coor[2]) / 2
        y_middle = (coor[1] + coor[3]) / 2
        # print("x_middle:",x_middle,"y_middle:",y_middle)
        mark=False
        for roi_tag in roi_tag_list:
            if (x_middle>roi_tag[2] or x_middle<roi_tag[0]) or (y_middle>roi_tag[3] or y_middle<roi_tag[1]) :
                mark=True
                break
        if mark:
            continue
        bboxes_list.append(coor)
        #640*640填充
        x_crop = coor[2] - coor[0]
        y_crop = coor[3] - coor[1]
        # print("x_crop:",x_crop,"y_crop:",y_crop)
        img_empty = np.zeros((640, 640, 3), np.uint8)
        img_empty[200:200 + y_crop, 200:200 + x_crop] += img_tag_[coor[1]:coor[3],coor[0]:coor[2]]

        # 二次识别
        bbox_cfg_result = inference_yolov5(yolo_rec, img_empty)
        bbox_cfg_result =check_iou(bbox_cfg_result,0.2)
        # print("bbox_cfg_result:",bbox_cfg_result)
        # 按横坐标排序组合结果
        label_list = [[item['label'], item['coor']] for item in bbox_cfg_result]
        label_list = sorted(label_list, key=lambda x: x[1][0], reverse=False)
        label=""
        for item in label_list:
            if str(item[0]).endswith('.0'):
                label+=str(item[0])[:-1]
            else:
                label+=str(item[0])
        value_list.append(label)
        s = (x_crop) / 50  # 根据框子大小决定字号和线条粗细。
        cv2.putText(img_tag_, str(label), (coor[2], coor[3]), cv2.FONT_HERSHEY_SIMPLEX, round(s), (0, 255, 0),
                    thickness=round(s * 2))
        cv2.rectangle(img_tag_, (int(coor[0]), int(coor[1])), (int(coor[2]), int(coor[3])), (255, 0, 255), thickness=1)

    # cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag_cfg.jpg"), img_tag_)
    out_data["msg"] = "Success!"
    out_data["code"] = 0
    out_data['data']['type'] = 'digital'
    out_data['data']['values'] = value_list
    out_data['data']['bboxes'] = bboxes_list

    ## 输出可视化结果的图片。
    # f = open(os.path.join(save_path, TIME_START + "output_data.json"), "w", encoding='utf-8')
    # json.dump(out_data, f, indent=2, ensure_ascii=False)
    # f.close()
    out_data["img_result"] = img2base64(img_tag_)
    return out_data




if __name__ == '__main__':
    img_list=os.listdir('test')
    for item in img_list:
        # if item.endswith(".jpg"):
        #     img_tag_file = "test/"+item
        #     img_tag = img2base64(cv2.imread(img_tag_file))
        #     input_data = {"image": img_tag, "config": {}, "type": "digital"}
        if item.endswith("input_data.json"):
            with open("test/"+item,"r",encoding="utf8") as f :
                input_data=json.load(f)
            # print(input_data)
            out_data = inspection_digital_rec(input_data)
            print(item,out_data['msg'])
            # print(out_data['data']['values'])
            print("==================================")