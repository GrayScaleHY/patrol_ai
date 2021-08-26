from flask import Flask, request, jsonify
from io import BytesIO
import base64
import json
# from app_meter_recognition import meter_rec, meter_loc
from app_inspection_pointer import inspection_pointer

app = Flask(__name__)

# 让jsonify返回的json串支持中文
app.config['JSON_AS_ASCII'] = False

@app.route('/meter_recognition/', methods=['POST'])
def meter_recognition():
    # 检测结果
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}  # 请求失败
        return jsonify(res)
        # 获取图片文件
    try:
        data = json.loads(request.get_data(as_text=True))
        res = meter_rec(data)
        return jsonify(res)
    except: #Exception as ex
        # print(ex)
        res = {'code': 1, 'msg': 'Request exception!','data': []}
        return jsonify(res)


@app.route('/meter_location/', methods=['POST'])
def meter_location():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}  # 请求失败
        return jsonify(res)
    try:
        data = json.loads(request.get_data(as_text=True))
        res = meter_loc(data)
        return jsonify(res)
    except:
        res = {'code': 1, 'msg': 'Request exception!','data': []}
        return jsonify(res)

@app.route('/inspection_pointer_server/', methods=['POST'])
def inspection_pointer_server():
    # 检测结果
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}  # 请求失败
        return jsonify(res)
    # 获取请求的json数据
    data = json.loads(request.get_data(as_text=True))
    res = inspection_pointer(data)
    print("inspection_pointer result:")
    print("-----------------------------------------------")
    print(res)
    print("----------------------------------------------")
    return jsonify(res)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
