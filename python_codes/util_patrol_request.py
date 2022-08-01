import argparse
import base64
import json
import requests
import lib_image_ops
import cv2
from flask import request,make_response

API = "http://192.168.44.135:5000/inspection_rec_defect/"

# img_file = "/home/yh/image/python_codes/inspection_result/digital/11-29-19-26-13/img_tag.jpg"
# img_base64 = lib_image_ops.img2base64(cv2.imread(img_file))
# input_data = {"image": img_base64, "config":[], "type": "digital"}
f = open("/data/PatrolAi/patrol_ai/python_codes/inspection_result/07-28-15-10-35/input_data.json","r",encoding='utf-8')
input_data = json.load(f)
f.close()

send_data = json.dumps(input_data)
res = requests.post(url=API, data=send_data).json()

print("------------------------------")
for s in res:
    if s != "img_result":
        print(s,":",res[s])
print("------------------------------")
