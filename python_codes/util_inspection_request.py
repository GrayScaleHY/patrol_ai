import argparse
import base64
import json
import requests
import lib_image_ops
import cv2
from flask import request,make_response

def requst_inspection_pointer():
    """
    智能巡视-指针读数。
    """
    API = "http://192.168.57.159:5000/inspection_pointer/"

    ## 输入参数
    img_ref_file = "/home/yh/image/python_codes/test/test1.jpg"
    img_tag_file = "/home/yh/image/python_codes/test/test1.jpg"
    coordinates = {"center": [1074, 609], "-0.1": [919, 796], "0": [854, 709], "0.2": [864, 513], "0.4": [1058, 392], "0.6": [1256, 476], "0.8": [1312, 670], "0.9": [1253, 759]}
    img_ref = cv2.imread(img_tag_file)
    W = img_ref.shape[1]; H = img_ref.shape[0]
    for t in coordinates:
        coor = [coordinates[t][0]/W, coordinates[t][1]/H]
        coordinates[t] = coor

    img_ref = lib_image_ops.img2base64(img_ref)
    img_tag = lib_image_ops.img2base64(cv2.imread(img_ref_file))
    
    input_data = {"image": img_tag, "config": {"img_ref": img_ref, "coordinates": coordinates}, "type": "pointer"}
    send_data = json.dumps(input_data)
    res = requests.post(url=API, data=send_data).json()
    print("------------------------------")
    for s in res:
        if s != "img_result":
            print(s,":",res[s])
    print("------------------------------")

    img = lib_image_ops.base642img(res["img_result"])
    cv2.imwrite(img_tag_file[:-4] + "_result.jpg", img)

def requst_inspection_disconnector():
    """
    智能巡视-指针读数。
    """
    API = "http://192.168.57.159:5000/inspection_disconnector/"

    tag_file = "images/test_0_open_open.png"
    open_file = "images/test_0_close_open.png"
    close_file = "images/test_0_open_close.png"
    bbox =  [1460, 405, 1573, 578]
    img_tag = lib_image_ops.img2base64(cv2.imread(tag_file))
    img_open = lib_image_ops.img2base64(cv2.imread(open_file))
    img_close = lib_image_ops.img2base64(cv2.imread(close_file))
    input_data = {
        "image": img_tag,
        "config": {"img_open": img_open, "img_close": img_close, "bbox": bbox},
        "type": "disconnector"
    }

    send_data = json.dumps(input_data)
    res = requests.post(url=API, data=send_data).json()

    print("------------------------------")
    for s in res:
        if s != "img_result":
            print(s,":",res[s])
    print("------------------------------")


def requst_inspection_counter():
    """
    智能巡视-计数器读数。
    """
    API = "http://192.168.57.159:5000/inspection_counter/"

    img_file = "/home/yh/image/python_codes/test/test1.jpg"
    img_base64 = lib_image_ops.img2base64(cv2.imread(img_file))
    input_data = {"image": img_base64, "config":[], "type": "counter"}

    send_data = json.dumps(input_data)
    res = requests.post(url=API, data=send_data).json()

    print("------------------------------")
    for s in res:
        if s != "img_result":
            print(s,":",res[s])
    print("------------------------------")


def requst_inspection_meter():
    """
    智能巡视-仪表定位。
    """
    API = "http://192.168.57.159:5000/inspection_meter/"

    img_file = "images/img_ref.jpg"
    img_base64 = lib_image_ops.img2base64(cv2.imread(img_file))
    input_data = {"image": img_base64, "config":[], "type": "meter"}

    send_data = json.dumps(input_data)
    res = requests.post(url=API, data=send_data).json()

    print("------------------------------")
    for s in res:
        if s != "img_result":
            print(s,":",res[s])
    print("------------------------------")

def requst_inspection_object_detection():
    """
    目标检测。
    """
    API = "http://192.168.57.159:5000/inspection_led/"

    img_file = "/data/yolov5/led/images/val/2020_8_4_led_82.jpg"
    img_base64 = lib_image_ops.img2base64(cv2.imread(img_file))
    input_data = {"image": img_base64, "config":[], "type": "led"}

    send_data = json.dumps(input_data)
    res = requests.post(url=API, data=send_data).json()

    print("------------------------------")
    for s in res:
        if s != "img_result":
            print(s,":",res[s])
    print("------------------------------")

if __name__ == '__main__':
    # requst_inspection_pointer()
    # requst_inspection_meter()
    requst_inspection_counter()
    # requst_inspection_disconnector()
    # requst_inspection_disconnector()