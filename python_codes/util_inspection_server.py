from flask import Flask, request, jsonify
import json
import os
from app_inspection_pointer import inspection_pointer
from app_inspection_meter import inspection_meter
from app_inspection_disconnector import inspection_disconnector
from app_inspection_counter import inspection_counter

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False # 让jsonify返回的json串支持中文

@app.route('/inspection_pointer/', methods=['POST'])
def inspection_pointer_server():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))
    res = inspection_pointer(data)
    print("inspection_pointer result:")
    print("-----------------------------------------------")
    print(res)
    print("----------------------------------------------")
    return jsonify(res)


@app.route('/inspection_meter/', methods=['POST'])
def inspection_meter_server():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))
    res = inspection_meter(data)
    print("meter_location result:")
    print("-----------------------------------------------")
    print(res)
    print("----------------------------------------------")
    return jsonify(res)


@app.route('/inspection_counter/', methods=['POST'])
def inspection_counter_server():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))
    res = inspection_counter(data)
    print("inspection_pointer result:")
    print("-----------------------------------------------")
    print(res)
    print("----------------------------------------------")
    return jsonify(res)


@app.route('/inspection_disconnector/', methods=['POST'])
def inspection_disconnector_server():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))
    res = inspection_disconnector(data)
    print("inspection_pointer result:")
    print("-----------------------------------------------")
    print(res)
    print("----------------------------------------------")
    return jsonify(res)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
