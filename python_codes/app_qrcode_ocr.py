import cv2
import os
import time
import json
import numpy as np
from lib_image_ops import base642img, img2base64, img_chinese
from lib_sift_match import sift_match, convert_coor, sift_create
from lib_qrcode import decoder, decoder_wechat
from lib_inference_ocr import load_ppocr, inference_ppocr
from lib_help_base import GetInputData, is_include
## 加载padpad模型
from config_load_models_var import text_sys

def decoder_qrcode(img, roi):
    img_roi = img[int(roi[1]): int(roi[3]), int(roi[0]): int(roi[2])]
    try:
        boxes = decoder_wechat(img)
    except:
        boxes = decoder(img)

    boxes = [b for b in boxes if is_include(b['bbox'], roi, srate=0.5)]

    if len(boxes) == 0:
        try:
            boxes = decoder_wechat(img_roi)
        except:
            boxes = decoder(img_roi)
        for i in range(len(boxes)):
            c = boxes[i]["bbox"]
            boxes[i]["bbox"] = [int(c[0]+roi[0]), int(c[1]+roi[1]), int(c[2]+roi[0]), int(c[3]+roi[1])]

    return boxes

def inspection_qrcode(input_data):
    """
    解二维码
    """

    ## 提取输入请求信息
    DATA = GetInputData(input_data)
    checkpoint = DATA.checkpoint; an_type = DATA.type
    img_tag = DATA.img_tag; img_ref = DATA.img_ref
    roi = DATA.roi; osd = DATA.osd

    out_data = {"code": 0, "data":[], "img_result": input_data["image"], "msg": "Request; "} 

    ## 画上点位名称和osd区域
    img_tag_ = img_tag.copy()
    img_tag_ = img_chinese(img_tag_, checkpoint + an_type , (10, 10), color=(255, 0, 0), size=60)
    for o_ in osd:  ## 如果配置了感兴趣区域，则画出osd区域
        cv2.rectangle(img_tag_, (int(o_[0]), int(o_[1])),(int(o_[2]), int(o_[3])), (255, 0, 255), thickness=1)
        cv2.putText(img_tag_, "osd", (int(o_[0]), int(o_[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)

    ## 求出目标图像的感兴趣区域
    if len(roi) > 0:
        if len(osd) == 0:
            osd = [[0,0,1,0.1],[0,0.9,1,1]]
        feat_ref = sift_create(img_ref, rm_regs=osd)
        feat_tag = sift_create(img_tag)
        M = sift_match(feat_ref, feat_tag, ratio=0.5, ops="Perspective")
        if M is None:
            out_data["msg"] = out_data["msg"] + "; Not enough matches are found"
            roi_tag = roi[0]
        else:
            roi = roi[0]
            coors = [(roi[0],roi[1]), (roi[2],roi[1]), (roi[2],roi[3]), (roi[0],roi[3])]
            coors_ = [list(convert_coor(coor, M)) for coor in coors]
            c_ = np.array(coors_, dtype=int)
            roi_tag = [min(c_[:,0]), min(c_[:, 1]), max(c_[:,0]), max(c_[:,1])]
    else:
        roi_tag = [0,0, img_tag.shape[1], img_tag.shape[0]]
    img_roi = img_tag[int(roi_tag[1]): int(roi_tag[3]), int(roi_tag[0]): int(roi_tag[2])]
    
    ## 画出roi_tag
    c = roi_tag
    cv2.rectangle(img_tag_, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), (255,0,0), thickness=1)
    cv2.putText(img_tag_, "roi", (int(c[0]), int(c[1]) + 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)

    ## 二维码检测或文本检测
    if an_type == "qrcode":
        boxes = decoder_qrcode(img_tag, roi_tag)
    elif "ocr" in an_type: # 文本检测
        boxes = inference_ppocr(img_roi, text_sys)
        for i in range(len(boxes)):
            c = boxes[i]["bbox"]; r = roi_tag
            boxes[i]["bbox"] = [int(c[0]+r[0]), int(c[1]+r[1]), int(c[2]+r[0]), int(c[3]+r[1])]
    else:
        out_data["msg"] = out_data["msg"] + "; Type is wrong !"
        out_data["code"] = 1
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
        out_data["img_result"] = img2base64(img_tag_)
        return out_data

    if len(boxes) == 0: #没有检测到目标
        out_data["msg"] = out_data["msg"] + "; Not find qrcode"
        out_data["code"] = 1
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
        out_data["img_result"] = img2base64(img_tag_)
        return out_data

    for box in boxes:
        cfg = {"type": an_type, "content": box["content"], "bbox": box["bbox"]}
        out_data["data"].append(cfg)

    ## 可视化计算结果
    s = (roi_tag[2] - roi_tag[0]) / 200 # 根据框子大小决定字号和线条粗细。
    cv2.rectangle(img_tag_, (int(roi_tag[0]), int(roi_tag[1])),
                    (int(roi_tag[2]), int(roi_tag[3])), (0, 0, 255), thickness=round(s*2))
    cv2.putText(img_tag_, "roi", (int(roi_tag[0]), int(roi_tag[1]-s)),
                    cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 255), thickness=round(s))
    for bbox in boxes:
        coor = bbox["bbox"]; label = bbox["content"]
        s = int((coor[2] - coor[0]) / 3) # 根据框子大小决定字号和线条粗细。
        cv2.rectangle(img_tag_, (int(coor[0]), int(coor[1])),
                    (int(coor[2]), int(coor[3])), (0, 225, 0), thickness=round(s/50))
        # cv2.putText(img, label, (int(coor[0])-5, int(coor[1])-5),
        img_tag_ = img_chinese(img_tag_, label, (coor[0], coor[1]-round(s)), color=(0, 225, 0), size=round(s))
    
    ## 输出可视化结果的图片。
    img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
    out_data["img_result"] = img2base64(img_tag_)

    return out_data


if __name__ == '__main__':
    import glob
    import time
    from lib_help_base import get_save_head, save_output_data,save_input_data
    
    # tag_file = "/data/PatrolAi/result_patrol/img_tag.jpg"
    
    # img_tag = img2base64(cv2.imread(tag_file))
    # img_ = cv2.imread(ref_file)
    # img_ref = img2base64(img_)
    # ROI = [907, 7, 1583, 685]
    # W = img_.shape[1]; H = img_.shape[0]
    # roi = [ROI[0]/W, ROI[1]/H, ROI[2]/W, ROI[3]/H]
    json_file = "/data/PatrolAi/result_patrol/0105164329_input_data.json"
    # input_data = {"image": img_tag, "config":{}, "type": "qrcode"} # "img_ref": img_ref, "bboxes": {"roi": roi}
    f = open(json_file,"r",encoding='utf-8')
    input_data = json.load(f)
    f.close()
    out_data = inspection_qrcode(input_data)
    save_dir, name_head = get_save_head(input_data)
    save_input_data(input_data, save_dir, name_head, draw_img=True)
    save_output_data(out_data, save_dir, name_head)
    print("inspection_qrcode result:")
    print("-----------------------------------------------")
    for s in out_data:
        if s != "img_result":
            print(s,":",out_data[s])
    print("----------------------------------------------")

