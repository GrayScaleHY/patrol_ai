from lib_inference_ocr import load_ppocr,inference_ppocr
import cv2
import glob
import numpy as np
from lib_image_ops import base642img, img2base64, img_chinese
from lib_sift_match import sift_match, convert_coor, sift_create , fft_registration
import time
import json
import os
from config_load_models_var import text_sys
from lib_help_base import GetInputData

# det_model_dir = "/data/PatrolAi/ppocr/ch_PP-OCRv2_det_infer/"
# cls_model_dir = "/data/PatrolAi/ppocr/ch_ppocr_mobile_v2.0_cls_infer/"
# rec_model_dir = "/data/PatrolAi/ppocr/ch_PP-OCRv2_rec_infer/"
# text_sys = load_ppocr(det_model_dir, cls_model_dir, rec_model_dir)

def deal_str(ocr_str):
    deal_ocr_str = ocr_str[:-1]
    deal_ocr_str = deal_ocr_str.replace('C','0')
    deal_ocr_str = deal_ocr_str.replace('U', '0')
    deal_ocr_str = deal_ocr_str.replace('D', '0')
    deal_ocr_str = deal_ocr_str.replace('Q', '0')
    deal_ocr_str = deal_ocr_str.replace('A', '4')
    deal_ocr_str = deal_ocr_str.replace('-', '.')
    deal_ocr_str = deal_ocr_str.replace('·', '.')
    deal_ocr_str = deal_ocr_str.replace(':', '.')
    deal_ocr_str = deal_ocr_str.replace(' ', '')
    if deal_ocr_str=='000':
        deal_ocr_str='0.00'
    if deal_ocr_str=='00':
        deal_ocr_str='0.0'
    if deal_ocr_str=='0.':
        deal_ocr_str = '0.0'
    deal_ocr_str = deal_ocr_str + 'A'
    if '.' not in deal_ocr_str and deal_ocr_str[0]=='0':
        deal_ocr_str = deal_ocr_str[0] + '.' + deal_ocr_str[1:]
    return deal_ocr_str

def ocr_digit_detection(input_data):

    ## 初始化输入输出信息。
    # img_tag, img_ref, roi = get_input_data(input_data)
    data = GetInputData(input_data)
    img_tag, img_ref, roi = data.img_tag, data.img_ref, data.roi
    checkpoint = data.checkpoint
    if data.roi==[]:
        roi=None
    else:
        roi=roi[0]
    
    out_data = {"code": 0, "data": {}, "img_result": data.img_tag,
                "msg": "Success request object detect; "}  # 初始化out_data

    ## 将输入请求信息可视化
    img_tag_ = img_tag.copy()
    img_tag_ = img_chinese(img_tag_, checkpoint + data.type , (10, 10), color=(255, 0, 0), size=60)

    if roi is not None:  # 如果配置了感兴趣区域，则画出感兴趣区域
        img_ref_ = img_ref.copy()
        cv2.rectangle(img_ref_, (int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])), (255, 0, 255), thickness=1)
        cv2.putText(img_ref_, "roi", (int(roi[0]), int(roi[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255),
                    thickness=1)

    if "augm" in data.config:
        if isinstance(data.config["augm"], list):
            if len(data.config["augm"]) == 2:
                augm = data.config["augm"]
                augm = [float(augm[0]), float(augm[1])]
                img_tag = np.uint8(np.clip((augm[0] * img_tag + augm[1]), 0, 255))


    ## 求出目标图像的感兴趣区域
    if roi is not None:
        # feat_ref = sift_create(img_ref, rm_regs=[[0, 0, 1, 0.1], [0, 0.9, 1, 1]])
        # feat_tag = sift_create(img_tag, rm_regs=[[0, 0, 1, 0.1], [0, 0.9, 1, 1]])
        # M = sift_match(feat_ref, feat_tag, ratio=0.5, ops="Perspective")
        M = fft_registration(img_ref, img_tag)
        if M is None:
            out_data["msg"] = out_data["msg"] + "; Not enough matches are found"
            roi_tag = roi
            xmin = max(0, min(roi[0], roi[2]))
            ymin = max(0, min(roi[1], roi[3]))
            xmax = min(img_tag.shape[1], max(roi[0], roi[2]))
            ymax = min(img_tag.shape[0], max(roi[1], roi[3]))
        else:
            coors = [(roi[0], roi[1]), (roi[2], roi[1]), (roi[2], roi[3]), (roi[0], roi[3])]
            coors_ = []
            for coor in coors:
                coors_.append(list(convert_coor(coor, M)))
            xs = [coor[0] for coor in coors_]
            ys = [coor[1] for coor in coors_]
            xmin = max(0, min(xs));
            ymin = max(0, min(ys))
            xmax = min(img_tag.shape[1], max(xs));
            ymax = min(img_tag.shape[0], max(ys))
            roi_tag = [xmin, ymin, xmax, ymax]

        ## 画出roi_tag
        c = roi_tag
        cv2.rectangle(img_tag_, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (255, 0, 255), thickness=1)
        cv2.putText(img_tag_, "roi", (int(c[0]), int(c[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255),
                    thickness=1)

        '''------------------------------------------
                             模型推理
        ---------------------------------------------'''
        img_tag_roi=img_tag[roi_tag[1]:roi_tag[3],roi_tag[0]:roi_tag[2]].copy()
        srcImg=img_tag_.copy()
        srcImg[:, :, :] = 0
        srcImg[ymin:ymax, xmin:xmax] = img_tag_roi
    else: ##没有roi
        srcImg=img_tag_.copy()
        roi_tag=[0,0,srcImg.shape[1]-1,srcImg.shape[0]-1]

    # from PIL import ImageEnhance
    # # 对比度增强
    # enh_con = ImageEnhance.Contrast(srcImg)
    # contrast = 3
    # srcImg = enh_con.enhance(contrast)

    result = inference_ppocr(srcImg, text_sys)
    print("------------------------------")
    print(result)
    print(roi_tag)

    ycenter=0.5*(roi_tag[1]+roi_tag[3])
    xmin,ymin,xmax,ymax = roi_tag[0],roi_tag[1],roi_tag[2],roi_tag[3]
    out_data["data"] = {"type": "digital",
                        "values": ['0.0A', '0.0A' ,'0.0A'],
                        "bboxes": [[int(xmin), int(ymin), int(xmax), int(0.667*ymin+0.333*ymax)],
                                   [int(xmin), int(0.667*ymin+0.333*ymax), int(xmax), int(0.333*ymin+0.667*ymax)],
                                   [int(xmin), int(0.333*ymin+0.667*ymax), int(xmax), int(ymax)]]}
    for res in result:
        bbox = res["bbox"]
        content = deal_str(res["content"])
        # cv2.rectangle(img_tag_, (int(bbox[0]), int(bbox[1])),
        #               (int(bbox[2]), int(bbox[3])), (0, 0, 255), thickness=2)
        # img_tag_ = img_chinese(img_tag_, deal_str(content), (int(bbox[0]), int(bbox[1] - 20)), (0, 0, 255), size=20)
        # if content=='0.00' or content=='0.0' or content=='0.00A' or content=='0.0A':
        #     for i in result:
        #         i["content"] = '0.0A'
        #     out_data["data"] = {"type": "digital",
        #                         "values": ['0.0A', '0.0A', '0.0A'],
        #                         "bboxes": [[int(xmin), int(ymin), int(xmax), int(0.667 * ymin + 0.333 * ymax)],
        #                                    [int(xmin), int(0.667 * ymin + 0.333 * ymax), int(xmax),
        #                                     int(0.333 * ymin + 0.667 * ymax)],
        #                                    [int(xmin), int(0.333 * ymin + 0.667 * ymax), int(xmax), int(ymax)]]}
                # if i["bbox"][1] < ycenter and i["bbox"][3] < ycenter:
                #     out_data["data"]['values'][0] = deal_str(i["content"])
                #     out_data["data"]['bboxes'][0] = [int(i["bbox"][0]), int(i["bbox"][1]), int(i["bbox"][2]), int(i["bbox"][3])]
                # elif i["bbox"][1] < ycenter and i["bbox"][3] > ycenter:
                #     out_data["data"]['values'][1] = deal_str(i["content"])
                #     out_data["data"]['bboxes'][1] = [int(i["bbox"][0]), int(i["bbox"][1]), int(i["bbox"][2]), int(i["bbox"][3])]
                # elif i["bbox"][1] > ycenter and i["bbox"][3] > ycenter:
                #     out_data["data"]['values'][2] = deal_str(i["content"])
                #     out_data["data"]['bboxes'][2] = [int(i["bbox"][0]), int(i["bbox"][1]), int(i["bbox"][2]), int(i["bbox"][3])]
        # else:
        if bbox[1]<ycenter and bbox[3]<ycenter:
            out_data["data"]['values'][0]=deal_str(content)
            out_data["data"]['bboxes'][0] = [int(bbox[0]), int(bbox[1]),int(bbox[2]), int(bbox[3])]
        elif bbox[1]<ycenter and bbox[3]>ycenter:
            out_data["data"]['values'][1]=deal_str(content)
            out_data["data"]['bboxes'][1] = [int(bbox[0]), int(bbox[1]),int(bbox[2]), int(bbox[3])]
        elif bbox[1]>ycenter and bbox[3]>ycenter:
            out_data["data"]['values'][2]=deal_str(content)
            out_data["data"]['bboxes'][2] = [int(bbox[0]), int(bbox[1]),int(bbox[2]), int(bbox[3])]

    # if len(result)<3:
    for i in range(3):
        cv2.rectangle(img_tag_, (int(out_data["data"]['bboxes'][i][0]), int(out_data["data"]['bboxes'][i][1])),
                      (int(out_data["data"]['bboxes'][i][2]), int(out_data["data"]['bboxes'][i][3])), (0, 0, 255), thickness=2)
        img_tag_ = img_chinese(img_tag_, deal_str(out_data["data"]['values'][i]), (int(out_data["data"]['bboxes'][i][0]), int(out_data["data"]['bboxes'][i][1] - 20)), (0, 0, 255), size=20)


    ## 输出可视化结果的图片。
    out_data["img_result"] = img2base64(img_tag_)

    return out_data

if __name__ == '__main__':
    json_file = "test.json"
    f = open(json_file, "r", encoding='utf-8')
    input_data = json.load(f)
    f.close()
    out_data = ocr_digit_detection(input_data)
    print("ocr_digit_detection result:")
    print("-----------------------------------------------")
    for s in out_data:
        if s != "img_result":
            print(s, ":", out_data[s])
    print("----------------------------------------------")
