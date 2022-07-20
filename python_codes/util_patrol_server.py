from flask import Flask, request, jsonify
import json
import os
from app_pointer import inspection_pointer
from app_disconnector import inspection_disconnector
from app_counter import inspection_counter
from app_object_detection import inspection_object_detection
from app_qrcode_ocr import inspection_qrcode
# from app_yeweiji import inspection_level_gauge
# from app_guijiaobianse import inspection_xishiqi
from app_panbie import inspection_identify_defect
from config_version import code_version
from app_disconnector_video import inspection_disconnector_video

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False # 让jsonify返回的json串支持中文

@app.route('/inspection_state/', methods=['POST'])
def inspection_state():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    else:
        res = {"code": 0}
    return jsonify(res)

@app.route('/inspection_version/', methods=['POST'])
def inspection_version():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    else:
        res = {"data":code_version, "code": 0}
    return jsonify(res)

## 刀闸分合识别
@app.route('/inspection_disconnector/', methods=['POST'])
def inspection_disconnector_server():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))
    res = inspection_disconnector(data)
    print("inspection_pointer result:")
    print("-----------------------------------------------")
    for s in res:
        if s != "img_result":
            print(s,":",res[s])
    print("----------------------------------------------")
    return jsonify(res)

## 视频分析刀闸分合状态
@app.route('/inspection_disconnector_video/', methods=['POST'])
def inspection_disconnector_video_server():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))
    res = inspection_disconnector_video(data)
    print("inspection_pointer result:")
    print("-----------------------------------------------")
    for s in res:
        if s != "img_result":
            print(s,":",res[s])
    print("----------------------------------------------")
    return jsonify(res)

## 仪表指针读数
@app.route('/inspection_pointer/', methods=['POST'])
def inspection_pointer_server():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))
    res = inspection_pointer(data)
    print("inspection_pointer result:")
    print("-----------------------------------------------")
    for s in res:
        if s != "img_result":
            print(s,":",res[s])
    print("----------------------------------------------")
    return jsonify(res)

## 判别算法
@app.route('/inspection_identify_defect/', methods=['POST'])
def inspection_identify_defect_server():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))
    res = inspection_identify_defect(data)
    print("inspection_identify_defect result:")
    print("-----------------------------------------------")
    for s in res:
        if s != "img_result":
            print(s,":",res[s])
    print("----------------------------------------------")
    return jsonify(res)

## 仪表计数器读数
@app.route('/inspection_digital/', methods=['POST'])
@app.route('/inspection_counter/', methods=['POST'])
def inspection_counter_server():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))
    res = inspection_counter(data)
    print("inspection_pointer result:")
    print("-----------------------------------------------")
    for s in res:
        if s != "img_result":
            print(s,":",res[s])
    print("----------------------------------------------")
    return jsonify(res)

## 仪表定位
# @app.route('/inspection_meter/', methods=['POST'])
# def inspection_meter_server():
#     if request.method != 'POST':
#         res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
#         return jsonify(res)
#     data = json.loads(request.get_data(as_text=True))
#     res = inspection_meter(data)
#     print("meter_location result:")
#     print("-----------------------------------------------")
#     for s in res:
#         if s != "img_result":
#             print(s,":",res[s])
#     print("----------------------------------------------")
#     return jsonify(res)

## 二维码识别，文本标识牌识别
@app.route('/inspection_qrcode/', methods=['POST'])
@app.route('/inspection_ocr/', methods=['POST'])
@app.route('/inspection_screen_ocr/', methods=['POST'])
def inspection_qrcode_server():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))
    res = inspection_qrcode(data)
    print("meter_location result:")
    print("-----------------------------------------------")
    for s in res:
        if s != "img_result":
            print(s,":",res[s])
    print("----------------------------------------------")
    return jsonify(res)

# @app.route('/inspection_level_gauge/', methods=['POST'])
# def inspection_level_gauge_server():
#     if request.method != 'POST':
#         res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
#         return jsonify(res)
#     data = json.loads(request.get_data(as_text=True))
#     res = inspection_level_gauge(data)
#     print("meter_location result:")
#     print("-----------------------------------------------")
#     for s in res:
#         if s != "img_result":
#             print(s,":",res[s])
#     print("----------------------------------------------")
#     return jsonify(res)

# @app.route('/inspection_xishiqi_color/', methods=['POST'])
# def inspection_xishiqi_color_server():
#     if request.method != 'POST':
#         res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
#         return jsonify(res)
#     data = json.loads(request.get_data(as_text=True))
#     res = inspection_xishiqi(data)
#     print("meter_location result:")
#     print("-----------------------------------------------")
#     for s in res:
#         if s != "img_result":
#             print(s,":",res[s])
#     print("----------------------------------------------")
#     return jsonify(res)

## 目标检测
@app.route('/inspection_pressplate/', methods=['POST'])
@app.route('/inspection_led/', methods=['POST'])
# @app.route('/inspection_fire_smoke/', methods=['POST'])
@app.route('/inspection_air_switch/', methods=['POST'])
@app.route('/inspection_fangpaiqi/', methods=['POST'])
# @app.route('/inspection_helmet/', methods=['POST'])
@app.route('/inspection_meter/', methods=['POST'])
# @app.route('/inspection_arrow/', methods=['POST'])
@app.route('/inspection_rotary_switch/', methods=['POST'])
@app.route('/inspection_door/', methods=['POST'])
@app.route('/inspection_key/', methods=['POST'])
# @app.route('/inspection_robot/', methods=['POST'])
@app.route('/inspection_rec_defect/', methods=['POST'])
def inspection_object():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))
    res = inspection_object_detection(data)
    print("meter_location result:")
    print("-----------------------------------------------")
    for s in res:
        if s != "img_result":
            print(s,":",res[s])
    print("----------------------------------------------")
    return jsonify(res)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
