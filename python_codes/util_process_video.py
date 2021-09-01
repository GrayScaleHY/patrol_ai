import cv2
import lib_image_ops
import json
import requests

API = "http://192.168.57.159:5000/inspection_pointer/"

img_ref_file = "images/pointer/test_0.jpg"
coordinates = {"center": [546, 361], "-30": [480, 460], "-10": [444, 423], "10": [428, 377], "30": [432, 330], "50": [456, 290], "70": [496, 258]}

img_r = cv2.imread(img_ref_file)
img_ref = lib_image_ops.img2base64(img_r)

## 建立视频对象
input_file = "images/pointer/test_0.mp4"
output_file = "images/pointer/test_0_result.mp4"
cap = cv2.VideoCapture(input_file)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps =cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(output_file,fourcc, fps, size)
while(cap.isOpened()):

    ret, frame = cap.read() # 逐帧读取

    if ret==True:

        ## 对图像就行处理
        img_tag = lib_image_ops.img2base64(frame)
        input_data = {"image": img_tag, "config": {"img_ref": img_ref, "coordinates": coordinates}, "type": "pointer"}
        send_data = json.dumps(input_data)
        res = requests.post(url=API, data=send_data).json()
        img_base64 = res["img_result"]
        img = lib_image_ops.base642img(img_base64)

        ## 保存和显示
        out.write(img)
        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

## 释放内存
cap.release()
out.release()
cv2.destroyAllWindows()