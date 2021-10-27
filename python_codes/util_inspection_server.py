from flask import Flask, request, jsonify
import json
import os
from app_inspection_pointer import inspection_pointer
# from app_inspection_meter import inspection_meter
from app_inspection_disconnector import inspection_disconnector
from app_inspection_counter import inspection_counter
from app_inspection_object_detection import inspection_object_detection
from app_inspection_qrcode import inspection_qrcode

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False # 让jsonify返回的json串支持中文

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

## 仪表计数器读数
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
@app.route('/inspection_qrcond/', methods=['POST'])
@app.route('/inspection_ocr/', methods=['POST'])
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

## 目标检测
@app.route('/inspection_pressplate/', methods=['POST'])
@app.route('/inspection_led/', methods=['POST'])
@app.route('/inspection_fire_smoke/', methods=['POST'])
@app.route('/inspection_air_switch/', methods=['POST'])
@app.route('/inspection_fangpaiqi/', methods=['POST'])
@app.route('/inspection_helmet/', methods=['POST'])
@app.route('/inspection_digital/', methods=['POST'])
@app.route('/inspection_meter/', methods=['POST'])
@app.route('/inspection_arrow/', methods=['POST'])
@app.route('/inspection_rotary_switch/', methods=['POST'])
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
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
