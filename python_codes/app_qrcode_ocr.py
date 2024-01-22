import cv2
import os
import time
import json
import numpy as np
from lib_image_ops import base642img, img2base64, img_chinese
from lib_img_registration import roi_registration
from lib_qrcode import decoder, decoder_wechat
from lib_inference_ocr import inference_ocr
from lib_help_base import GetInputData, is_include


def decoder_qrcode_ocr(img, roi, infer_type="ocr"):
    """
    巡视算法解二维码
    args: 
        roi: 模板框，格式： {"roi_name": [xmin, ymin, xmax, ymax]}
    return:
        cfgs:输出分析类容，格式：{"no_roi": [{'bbox': [xmin,ymin,xmax,ymax], 'content':content}, ..]}}
    """
    data = {}
    for name, _roi in roi.items():
        img_roi = img[int(_roi[1]): int(_roi[3]), int(_roi[0]): int(_roi[2])]
        if infer_type == "ocr":
            cfgs = inference_ocr(img_roi)
        else:
            try:
                cfgs = decoder_wechat(img_roi)
            except:
                cfgs = decoder(img_roi)
        
        for i, cfg in cfgs:
            c = cfg[i]["bbox"]
            cfg[i]["bbox"] = [int(c[0]+roi[0]), int(c[1]+roi[1]), int(c[2]+roi[0]), int(c[3]+roi[1])]

        data[name] == cfgs
    
    return data

def inspection_qrcode(input_data):
    """
    解二维码
    """

    ## 提取输入请求信息
    DATA = GetInputData(input_data)
    checkpoint = DATA.checkpoint; an_type = DATA.type
    img_tag = DATA.img_tag; img_ref = DATA.img_ref
    roi = DATA.roi; osd = DATA.osd

    ## 初始化
    out_data = {"code": 1, "data":{}, "img_result": input_data["image"], "msg": "Request; "} 

    ## 画上点位名称
    img_tag_ = img_tag.copy()
    img_tag_ = img_chinese(img_tag_, an_type + "_" + checkpoint , (10, 100), color=(255, 0, 0), size=30)

    ## 求出目标图像的感兴趣区域
    img_tag_ = img_tag.copy()
    roi_tag = roi_registration(img_ref, img_tag, roi)
    for name, c in roi_tag.items():
        cv2.rectangle(img_tag_, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), (255,0,255), thickness=1)
        cv2.putText(img_tag_, name, (int(c[0]), int(c[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)
    
    ## 二维码检测或文本检测
    data = decoder_qrcode_ocr(img_tag, roi, infer_type=an_type)
    out_data["data"] = data

    ## 画出二维码
    for name, cfgs in data.items():
        for cfg in cfgs:
            c = cfg["bbox"]; content=cfg["content"]
            out_data["code"] = 0
            cv2.rectangle(img_tag_, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), (0,0,255), thickness=2)
            s = int((c[2] - c[0]) / 10) # 根据框子大小决定字号和线条粗细。
            img_tag_ = img_chinese(img_tag_, content, (c[0], c[1]), color=(0,0,255), size=s)

    ## 老版本的接口输出，"data"由字典改为list
    no_roi = [name.startswith("no_roi") for name in out_data["data"]]
    if all(no_roi): ## 全为1， 返回True
        _cfgs = []
        for name, _cfg in out_data["data"].items():
            _cfgs.append(_cfg[0])
        out_data["data"] = _cfgs
        if out_data["data"] == [{}]:
            out_data["data"] = []
    
    ## 输出可视化结果的图片。
    img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 130), color=(255, 0, 0), size=30)
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

