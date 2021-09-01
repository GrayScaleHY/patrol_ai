from flask import Flask, request, jsonify
import json
from app_meter_recognition import meter_loc, meter_rec

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False # 让jsonify返回的json串支持中文

@app.route('/meter_location/', methods=['POST'])
def meter_location():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))
    res = meter_loc(data)
    print("meter_location result:")
    print("-----------------------------------------------")
    print(res)
    print("----------------------------------------------")
    return jsonify(res)


@app.route('/meter_recognition/', methods=['POST'])
def meter_recognition():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))
    res = meter_rec(data)
    print("meter_rec result:")
    print("-----------------------------------------------")
    print(res)
    print("----------------------------------------------")
    return jsonify(res)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
