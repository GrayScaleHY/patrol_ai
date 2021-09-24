import cv2
import os
import time
import json
import numpy as np
from pyzbar.pyzbar import decode # sudo apt-get install libzbar-dev; pip install pyzbar
from lib_image_ops import base642img, img2base64

def decoder(img):
    """
    二维码定位，并且读取二维码信息
    https://towardsdatascience.com/build-your-own-barcode-and-qrcode-scanner-using-python-8b46971e719e
    args:
        img: image data
    return:
        info: 格式为: [{'bbox': [xmin,ymin,xmax,ymax], 'content':content, "c_type":c_type}, ..]
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
        bbox = [x, y, x+w, y+h]
        
        ## 取二维码的内容合类型
        content = str(obj.data.decode("utf-8"))
        c_type = str(obj.type)

        info.append({'bbox':bbox, "content": content, "c_type": c_type})

    return info


def inspection_qrcode(input_data):
    """
    解二维码
    """
    ## 初始化输入输出信息。
    out_data = {"code": 0, "data":[], "img_result": "image", "msg": "Success request object detect; "} # 初始化out_data

    ## 可视化输入信息
    TIME_START = time.strftime("%m-%d-%H-%M-%S") 
    save_path = os.path.join("inspection_result", input_data["type"], TIME_START)
    os.makedirs(save_path, exist_ok=True)
    f = open(os.path.join(save_path, "input_data.json"), "w")
    json.dump(input_data, f, ensure_ascii=False)  # 保存输入信息json文件
    f.close()
    img = base642img(input_data["image"])
    cv2.imwrite(os.path.join(save_path, "img_ref.jpg"), img) # 将输入图片可视化

    info = decoder(img) # 解二维码

    if len(info) == 0:
        out_data["msg"] = out_data["msg"] + "; Not find qrcode"
        return out_data

    for inf in info:
        coor = inf["bbox"]
        content = inf["content"]
        out_data["data"].append({"content": content, "bbox": coor})
        s = (coor[2] - coor[0]) / 200 # 根据框子大小决定字号和线条粗细。
        cv2.rectangle(img, (int(coor[0]), int(coor[1])),
                        (int(coor[2]), int(coor[3])), (0, 0, 255), thickness=round(s*2))
        cv2.putText(img, content, (int(coor[0])-5, int(coor[1])-5),
                        cv2.FONT_HERSHEY_SIMPLEX, s/2, (0, 0, 255), thickness=round(s))

    ## 输出可视化结果的图片。
    out_data["img_result"] = img2base64(img)

    return out_data

