import argparse
import base64
import json
import requests
from flask import request,make_response

# 请求服务器的地址，固定的
API_recognition = "http://192.168.57.159:5000/meter_recognition/"
API_location = "http://192.168.57.159:5000/meter_location/"


input_img = '/home/yh/meter_recognition/test/arrester/07-27-09-07-49/raw_image.jpg'
# img_raw = cv2.imread(img_file)
with open(input_img, "rb") as imageFile:
    img = imageFile.read()
image = base64.b64encode(img).decode('utf-8')
data = {"image": image, "config": [{"141.7": 0,"35.49": 2.0,"60.4": 1.5}], "type": ["pointer", "counter"]}

send_data = json.dumps(data)

res = requests.post(url=API_recognition, data=send_data).json()
print(res)
print("----------------------------------------")

# res = requests.post(url=API_location, data=send_data).json()
# print(res)
# print("-------------------------------")



