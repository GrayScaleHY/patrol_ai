import argparse
import base64
import json
import requests
import lib_image_ops
from flask import request,make_response

def requst_inspection_pointer():
    """
    智能巡视-指针读数。
    """
    API = "http://192.168.57.159:5000/inspection_pointer/"

    img_ref_file = "images/img_ref.jpg"
    img_tag_file = "images/img_tag.jpg"
    coordinates = {"center": [1113, 476], "-0.1": [890, 779], "0.1": [740, 479], "0.2": [773, 313], "0.3": [887, 178], 
                "0.4": [1050, 101], "0.5": [1230, 106], "0.6": [1390, 191], "0.7": [1488, 349], "0.8": [1502, 531], "0.9": [1423, 701]}
    img_tag = lib_image_ops.img2base64(img_tag_file)
    img_ref = lib_image_ops.img2base64(img_ref_file)
    input_data = {"image": img_tag, "config": {"img_ref": img_ref, "coordinates": coordinates}, "type": "pointer"}

    send_data = json.dumps(input_data)
    res = requests.post(url=API, data=send_data).json()

    print("------------------------------")
    print(res)
    print("------------------------------")


def requst_inspection_meter():
    """
    智能巡视-指针读数。
    """
    API = "http://192.168.57.159:5000/inspection_meter/"

    img_file = "images/img_ref.jpg"
    img_base64 = lib_image_ops.img2base64(img_file)
    input_data = {"image": img_base64, "config":[], "type": "meter"}

    send_data = json.dumps(input_data)
    res = requests.post(url=API, data=send_data).json()

    print("------------------------------")
    print(res)
    print("------------------------------")


def requst_inspection_counter():
    """
    智能巡视-指针读数。
    """
    API = "http://192.168.57.159:5000/inspection_counter/"

    img_file = "images/#0389_MaxZoom.jpg"
    img_base64 = lib_image_ops.img2base64(img_file)
    input_data = {"image": img_base64, "config":[], "type": "counter"}

    send_data = json.dumps(input_data)
    res = requests.post(url=API, data=send_data).json()

    print("------------------------------")
    print(res)
    print("------------------------------")


def requst_inspection_disconnector():
    """
    智能巡视-指针读数。
    """
    API = "http://192.168.57.159:5000/inspection_disconnector/"

    tag_file = "images/test_0_open_open.png"
    open_file = "images/test_0_close_open.png"
    close_file = "images/test_0_open_close.png"
    bbox =  [1460, 405, 1573, 578]
    with open(tag_file, "rb") as imageFile:
        img_tag= imageFile.read()
    img_tag = base64.b64encode(img_tag).decode('utf-8')
    with open(close_file, "rb") as imageFile:
        img_close = imageFile.read()
    img_close = base64.b64encode(img_close).decode('utf-8')
    with open(open_file, "rb") as imageFile:
        img_open = imageFile.read()
    img_open = base64.b64encode(img_open).decode('utf-8')
    input_data = {
        "image": img_tag,
        "config": {"img_open": img_open, "img_close": img_close, "bbox": bbox},
        "type": "disconnector"
    }

    send_data = json.dumps(input_data)
    res = requests.post(url=API, data=send_data).json()

    print("------------------------------")
    print(res)
    print("------------------------------")

if __name__ == '__main__':
    # requst_inspection_pointer()
    # requst_inspection_meter()
    # requst_inspection_counter()
    requst_inspection_disconnector()