import numpy as np
import requests
import math

W = 704
H = 576

## 获取摄像机参数
get_url = "http://192.168.52.66:31010/api/v1/channel/getGisInfo" # ?chnid=1669
data = {"chnid": 1669}
cam_info = requests.get(get_url, params=data).json()
print(cam_info)

P = cam_info["data"]["PanPos"]
T = cam_info["data"]["TiltPos"]
Z = cam_info["data"]["ZoomPos"]
FOV_H = cam_info["data"]["Horizontal"]
FOV_V = cam_info["data"]["Vertical"]

# ## 基于Z=1时的视场角求Z=其他时的视场角
# FOV_H_1 = 51.349998474121094 # Z=1时的视场角
# FOV_V_1 = 30.260000228881836
# Z_1 = 1
# FOV_H = np.rad2deg(2 * math.atan(math.tan(np.deg2rad(FOV_H_1)/2)/(Z/Z_1)))
# FOV_V = np.rad2deg(2 * math.atan(math.tan(np.deg2rad(FOV_V_1)/2)/(Z/Z_1)))

pos_sdk = {'pan': P, 'tilt':T, 'zoom':Z}
print("raw PTZ:", pos_sdk)

## 计算将(x, y)像素居中的PTZ值
x = 704
y = 0
print("raw (x, y):", (x, y))
# 水平方向
if x > (W / 2):
    delt_x = np.rad2deg(np.arctan((x - W / 2) / (W / 2) * np.tan( np.deg2rad(FOV_H /2))))
else:
    delt_x = -np.rad2deg(np.arctan((W / 2 - x) / (W / 2) * np.tan( np.deg2rad(FOV_H /2))))
# 垂直方向
if y > (H / 2):
    delt_y = np.rad2deg(np.arctan((y - H / 2) / (H / 2) * np.tan( np.deg2rad(FOV_V/ 2))))
else:
    delt_y = -np.rad2deg(np.arctan((H / 2 - y) / (H / 2) * np.tan( np.deg2rad(FOV_V/ 2))))
# print("delt:[%.3f, %.3f]" % (delt_x, delt_y))
p = P + delt_x
print("out p:", p)
if p < 0:
    p = 0
elif p > 360:
    p = 360
t = T + delt_y
print("out t:", t)
if t < 0:
    t = 0
elif t > 90:
    t = 90

pos_sdk['pan'] = p        # 得到转换后的pt
pos_sdk['tilt'] += t      # 相机是倒置放置的。。。
print("target PTZ:", pos_sdk)

data = {"z": Z, "p": p, "t": t, "chnid": 4969}
put_url = "http://192.168.44.217:31010/api/v1/channel/ptzPos"
cam_put = requests.put(put_url, params=data).json()