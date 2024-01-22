import cv2 ## pip install opencv-python==4.5.2.52; pip install opencv-contrib-python==4.5.2.52
from pyzbar.pyzbar import decode # sudo apt-get install libzbar-dev; pip install pyzbar
import numpy as np
import os

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
        bbox = [float(x), float(y), float(x+w), float(y+h)]
        
        ## 取二维码的内容合类型
        content = str(obj.data.decode("utf-8"))
        c_type = str(obj.type)

        info.append({"type": "qrcode", "content": content, "bbox": bbox})

    return info

def decoder_wechat(img):
    """
    微信团队的二维码定位，并且读取二维码信息
    http://www.juzicode.com/opencv-note-wechat-qrcode-detect-decode/
    args:
        img: image data
    return:
        info: 格式为: [{'bbox': [xmin,ymin,xmax,ymax], 'content':content}, ..]
    """
    ## 二维码识别模型存放路线
    de_txt = '/data/PatrolAi/opencv_3rdparty/detect.prototxt'
    de_model = '/data/PatrolAi/opencv_3rdparty/detect.caffemodel'
    sr_txt = '/data/PatrolAi/opencv_3rdparty/sr.prototxt'
    sr_model = '/data/PatrolAi/opencv_3rdparty/sr.caffemodel'

    ## 识别二维码类容
    if os.path.exists(de_txt) and os.path.exists(de_model) and os.path.exists(sr_txt) and os.path.exists(sr_model):
        detect_obj = cv2.wechat_qrcode_WeChatQRCode(de_txt, de_model, sr_txt, sr_model) 
    else:
        print("warning: qrcode models is not exist !!")
        detect_obj = cv2.wechat_qrcode_WeChatQRCode()
    res, points = detect_obj.detectAndDecode(img) 

    ## 后处理
    info = []
    for i in range(len(points)):
        p = points[i].astype(int)
        content = res[i]
        bbox = [float(np.min(p[:,0])), float(np.min(p[:,1])), float(np.max(p[:,0])), float(np.max(p[:,1]))]
        info.append({"type": "qrcode", "content": content, "bbox": bbox})

    return info

if __name__ == '__main__':
    img = cv2.imread("/home/yh/image/python_codes/test/test/img_tag.jpg")
    info = decoder_wechat(img)
    print(info)