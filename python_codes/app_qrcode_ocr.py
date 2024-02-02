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
        if infer_type == "ocr": # ocr推理
            _cfgs = inference_ocr(img)
            cfgs = []
            for i, cfg in enumerate(_cfgs):
                if is_include(cfg["bbox"], _roi, 0.5):
                    cfgs.append(cfg)

        else: # 二维码推理
            img_roi = img[int(_roi[1]): int(_roi[3]), int(_roi[0]): int(_roi[2])]
            try:
                cfgs = decoder_wechat(img_roi)
            except:
                cfgs = decoder(img_roi)
            for i, cfg in enumerate(cfgs):
                c = cfg["bbox"]
                cfgs[i]["bbox"] = [int(c[0]+_roi[0]), int(c[1]+_roi[1]), int(c[2]+_roi[0]), int(c[3]+_roi[1])]

        data[name] = cfgs
    
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
    roi_tag, _ = roi_registration(img_ref, img_tag, roi)
    for name, c in roi_tag.items():
        cv2.rectangle(img_tag_, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), (255,0,255), thickness=1)
        cv2.putText(img_tag_, name, (int(c[0]), int(c[1])+10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)
    
    ## 二维码检测或文本检测
    data = decoder_qrcode_ocr(img_tag, roi_tag, infer_type=an_type)
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
    no_roi = [name.startswith("old_roi") for name in out_data["data"]]
    if all(no_roi): ## 全为1， 返回True
        cfgs = []
        for name, _cfgs in out_data["data"].items():
            for _cfg in _cfgs:
                cfgs.append(_cfg)
        out_data["data"] = cfgs
        if out_data["data"] == [{}]:
            out_data["data"] = []
    
    ## 输出可视化结果的图片。
    img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 130), color=(255, 0, 0), size=30)
    
    if os.path.exists(input_data["image"]): 
        out_file = input_data["image"][:-4] + "_result.jpg"
        cv2.imwrite(out_file, img_tag_)
        out_data["img_result"] = out_file
    else:
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
    json_file = "/data/PatrolAi/result_patrol/ocr/0123074334_文本标识牌识别_input_data.json"
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

