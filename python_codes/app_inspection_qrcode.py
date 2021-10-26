import cv2
import os
import time
import json
import numpy as np
from pyzbar.pyzbar import decode # sudo apt-get install libzbar-dev; pip install pyzbar
from lib_image_ops import base642img, img2base64, img_chinese
from app_inspection_disconnector import sift_match, convert_coor

def decoder(img):
    """
    二维码定位，并且读取二维码信息
    https://towardsdatascience.com/build-your-own-barcode-and-qrcode-scanner-using-python-8b46971e719e
    args:
        img: image data
    return:
        info: 格式为: [{'coor': [xmin,ymin,xmax,ymax], 'content':content, "c_type":c_type}, ..]
    """
    ## 彩图变灰度图
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    barcode = decode(img) # 解二维码

    info = []
    for obj in barcode:
        ## 取二维码的四边形框
        # points = obj.polygon
        # pts = np.array(points, np.int32)
        # pts = pts.reshape((-1, 1, 2))
        # cv2.polylines(img, [pts], True, (0, 255, 0), 3)

        ## 取二维码的矩形框
        (x,y,w,h) = obj.rect
        bbox = [float(x), float(y), float(x+w), float(y+h)]
        
        ## 取二维码的内容合类型
        content = str(obj.data.decode("utf-8"))
        c_type = str(obj.type)

        info.append({"coor": bbox, "content": content, "c_type": c_type})

    return info


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
    roi = None # 初始假设
    if "bboxes" in input_data["config"]:
        if isinstance(input_data["config"]["bboxes"], dict):
            if "roi" in input_data["config"]["bboxes"]:
                if isinstance(input_data["config"]["bboxes"]["roi"], list):
                    if len(isinstance(input_data["config"]["bboxes"]["roi"])) == 4:
                        W = img_ref.shape[1]; H = img_ref.shape[0]
                        roi = input_data["config"]["bboxes"]["roi"]
                        roi = [int(roi[0]*W), int(roi[1]*H), int(roi[2]*W), int(roi[3]*H)]   
    
    return img_tag, img_ref, roi


def inspection_qrcode(input_data):
    """
    解二维码
    """
    ## 初始化输入输出信息。
    ## 初始化输入输出信息。
    img_tag, img_ref, roi = get_input_data(input_data)
    out_data = {"code": 0, "data":[], "img_result": "image", "msg": "Success request object detect; "} # 初始化out_data

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
        roi_tag = [0,0, img_tag.shape[1], img_tag.shape[0]]
    else:
        coors = [(roi[0],roi[1]), (roi[2],roi[1]), (roi[2],roi[3]), (roi[0],roi[3])]
        coors_ = []
        for coor in coors:
            coors_.append(list(convert_coor(coor, M)))
        coors_ = np.array(coors_, dtype=int)
        roi_tag = [np.min(coors_[:,0]), np.min(coors_[:,1]), np.max(coors_[:,0]), np.max(coors_[:,1])]
    img_roi = img_tag[int(roi_tag[1]): int(roi_tag[3]), int(roi_tag[0]): int(roi_tag[2])]
    
    boxes = decoder(img_roi) # 解二维码
    if len(boxes) == 0: #没有检测到目标
        out_data["msg"] = out_data["msg"] + "; Not find qrcode"
        return out_data

    ## 将bboxes映射到原图坐标
    bboxes = []
    for bbox in boxes:
        c = bbox["coor"]; r = roi_tag
        coor = [c[0]+r[0], c[1]+r[1], c[2]+r[0], c[3]+r[1]]
        # {"coor": bbox, "content": content, "c_type": c_type}
        bboxes.append({"content": bbox["content"], "coor": coor, "c_type": bbox["c_type"]})

    for bbox in bboxes:
        cfg = {"type": "qrcode", "content": bbox["content"], "bbox": bbox["coor"]}
        out_data["data"].append(cfg)

    ## 可视化计算结果
    f = open(os.path.join(save_path, "out_data.json"), "w")
    json.dump(out_data, f, ensure_ascii=False, indent=2)  # 保存输入信息json文件
    f.close()
    s = (roi_tag[2] - roi_tag[0]) / 200 # 根据框子大小决定字号和线条粗细。
    cv2.rectangle(img_tag_, (int(roi_tag[0]), int(roi_tag[1])),
                    (int(roi_tag[2]), int(roi_tag[3])), (0, 0, 255), thickness=round(s*2))
    cv2.putText(img_tag_, "roi", (int(roi_tag[0]), int(roi_tag[1]-s)),
                    cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 255), thickness=round(s))
    for bbox in bboxes:
        coor = bbox["coor"]; label = bbox["content"]
        s = int((coor[2] - coor[0]) / 3) # 根据框子大小决定字号和线条粗细。
        cv2.rectangle(img_tag_, (int(coor[0]), int(coor[1])),
                    (int(coor[2]), int(coor[3])), (0, 225, 0), thickness=round(s/50))
        # cv2.putText(img, label, (int(coor[0])-5, int(coor[1])-5),
        img_tag_ = img_chinese(img_tag_, label, (coor[0], coor[1]-round(s/3)), color=(0, 225, 0), size=round(s/3))
    cv2.imwrite(os.path.join(save_path, "img_tag_cfg.jpg"), img_tag_)

    ## 输出可视化结果的图片。
    out_data["img_result"] = img2base64(img_tag_)

    return out_data


if __name__ == '__main__':
    tag_file = "test/16323889787350.png"
    ref_file = "test/p2vm1.jpg"
    img_tag = img2base64(cv2.imread(tag_file))
    img_ = cv2.imread(ref_file)
    img_ref = img2base64(img_)
    ROI = [907, 7, 1583, 685]
    W = img_.shape[1]; H = img_.shape[0]
    roi = [ROI[0]/W, ROI[1]/H, ROI[2]/W, ROI[3]/H]

    input_data = {"image": img_tag, "config":{"img_ref": img_ref, "bboxes": {"roi": roi}}, "type": "qrcode"} 
    out_data = inspection_qrcode(input_data)
    print("inspection_qrcode result:")
    print("-----------------------------------------------")
    for s in out_data:
        if s != "img_result":
            print(s,":",out_data[s])
    print("----------------------------------------------")

