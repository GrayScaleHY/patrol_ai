from flask import Flask, request, jsonify
import json
from lib_help_base import get_save_head, save_input_data, save_output_data, rm_result_patrolai
import os
from app_pointer import inspection_pointer
from app_disconnector import inspection_disconnector
# from app_counter import inspection_counter
from app_object_detection import inspection_object_detection
from app_qrcode_ocr import inspection_qrcode
from app_panbie import inspection_identify_defect
from config_version import code_version
from app_disconnector_video import inspection_disconnector_video
from app_yeweiji import inspection_level_gauge
# from app_ocr import ocr_digit_detection
from app_yejingpingshuzishibie import inspection_digital_rec
from app_match import patrol_match
import time
import threading
from config_object_name import AI_FUNCTION, convert_ai_function

## 单独开个进程，定时删除result_patrolai文件夹中的文件。
t = threading.Thread(target=rm_result_patrolai,args=())
t.start()

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

draw_img = True

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False # 让jsonify返回的json串支持中文

## ai能力列表
@app.route('/AI_function/', methods=['GET'])
def inspection_ai_function():
    return jsonify(convert_ai_function(AI_FUNCTION))

@app.route('/inspection_counter/', methods=['POST'])
@app.route('/inspection_digital/', methods=['POST'])
def inspection_digital_server():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))
    
    start_time = time.time()
    save_dir, name_head = get_save_head(data)
    save_input_data(data, save_dir, name_head, draw_img=draw_img)
    res = inspection_digital_rec(data)
    save_output_data(res, save_dir, name_head)

    print("-----------------------------------------------")
    for s in res:
        if s != "img_result":
            print(s,":",res[s])
    print("total spend time:", time.time() - start_time)
    print("----------------------------------------------")
    return jsonify(res)

@app.route('/inspection_track/', methods=['POST'])
@app.route('/inspection_track_batch/', methods=['POST'])
def inspection_track_server():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))
    
    start_time = time.time()
    save_dir, name_head = get_save_head(data)
    save_input_data(data, save_dir, name_head, draw_img=False)
    res = patrol_match(data)
    # save_output_data(res, save_dir, name_head)

    print("-----------------------------------------------")
    for s in res:
        if s != "img_result":
            print(s,":",res[s])
    print("total spend time:", time.time() - start_time)
    print("----------------------------------------------")
    return jsonify(res)


## 液位仪表读数
@app.route('/inspection_level_gauge/', methods=['POST'])
def inspection_level_gauge_server():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))

    start_time = time.time()
    save_dir, name_head = get_save_head(data)
    save_input_data(data, save_dir, name_head, draw_img=draw_img)
    res = inspection_level_gauge(data)
    save_output_data(res, save_dir, name_head)

    print("-----------------------------------------------")
    for s in res:
        if s != "img_result":
            print(s,":",res[s])
    print("total spend time:", time.time() - start_time)
    print("----------------------------------------------")
    return jsonify(res)

## 仪表指针读数
@app.route('/inspection_pointer/', methods=['POST'])
def inspection_pointer_server():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))

    start_time = time.time()
    save_dir, name_head = get_save_head(data)
    save_input_data(data, save_dir, name_head, draw_img=draw_img)
    res = inspection_pointer(data)
    save_output_data(res, save_dir, name_head)

    print("----------------------------------------------")
    for s in res:
        if s != "img_result":
            print(s,":",res[s])
    print("total spend time:", time.time() - start_time)
    print("----------------------------------------------")
    return jsonify(res)

## 二维码识别，文本标识牌识别
@app.route('/inspection_qrcode/', methods=['POST'])
@app.route('/inspection_ocr/', methods=['POST'])
def inspection_qrcode_server():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))

    start_time = time.time()
    save_dir, name_head = get_save_head(data)
    save_input_data(data, save_dir, name_head, draw_img=draw_img)
    res = inspection_qrcode(data)
    save_output_data(res, save_dir, name_head)

    print("-----------------------------------------------")
    for s in res:
        if s != "img_result":
            print(s,":",res[s])
    print("total spend time:", time.time() - start_time)
    print("----------------------------------------------")
    return jsonify(res)

## 目标检测
@app.route('/inspection_pressplate/', methods=['POST']) # 压板
@app.route('/inspection_led/', methods=['POST']) # led灯
@app.route('/inspection_led_color/', methods=['POST']) # led灯
@app.route('/inspection_fire_smoke/', methods=['POST']) # 烟火
@app.route('/inspection_air_switch/', methods=['POST']) # 空气开关
@app.route('/inspection_fangpaiqi/', methods=['POST']) # 翻牌器
@app.route('/inspection_helmet/', methods=['POST']) # 安全帽
@app.route('/inspection_meter/', methods=['POST']) # 表盘
@app.route('/inspection_rotary_switch/', methods=['POST']) # 旋钮开关
@app.route('/inspection_door/', methods=['POST']) # 箱门
@app.route('/inspection_key/', methods=['POST']) # 钥匙
@app.route('/inspection_rec_defect/', methods=['POST']) # 识别缺陷
# @app.route('/inspection_digital/', methods=['POST'])
# @app.route('/inspection_counter/', methods=['POST'])
@app.route('/inspection_person/', methods=['POST'])
@app.route('/inspection_disconnector_texie/', methods=['POST'])
@app.route('/inspection_disconnector_notemp/', methods=['POST'])
def inspection_object():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))
    
    start_time = time.time()
    save_dir, name_head = get_save_head(data)
    save_input_data(data, save_dir, name_head, draw_img=draw_img)
    res = inspection_object_detection(data)
    save_output_data(res, save_dir, name_head)

    print("-----------------------------------------------")
    for s in res:
        if s != "img_result":
            print(s,":",res[s])
    print("total spend time:", time.time() - start_time)
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
    print("-----------------------------------------------")
    for s in res:
        if s != "img_result":
            print(s,":",res[s])
    print("----------------------------------------------")
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

@app.route('/inspection_state/', methods=['POST'])
def inspection_state():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    else:
        res = {"code": 0}
    return jsonify(res)

#算法版本获取接口
@app.route('/historyVersion/', methods=['POST'])
def historyVersion():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    else:
        # 返回的数据
        res = {'count': 1,  #返回结果条数
               'data': {'algorithmManufacturer': 'UT', #算法厂商
                        'Version': "1.1.040715_release", #版本ID
                        'RecordTime':"2023.08.15", #该版本记录时间
                        'ModelDesc': 'AAA'}, #模型应用场景、评价指标等描述
               'message': 'AAA', #一般配合code使用，对异常/错误进行详细描述。
               'code': '200 OK'} #状态码，200 OK：成功;400 Bad Request：表示客户端请求有语法错误，不能被服务器所理解;401 Unauthonzed：表示请求未经授权;500：服务端异常。
    return jsonify(res)

#算法参数获取接口
@app.route('/recogParams/', methods=['POST'])
def recogParams():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    else:
        res = {'count': 1, #返回结果条数
               'data': {'sectionId': '666666', #区域编码
                        'sectionName': 'zhuhai', #区域名称
                        'stationId': '777777', #变电站编码
                        'stationName': 'zhuhaibiandianzhan', #变电站名称
                        'algorithmManufacturer': 'UT', #算法厂商
                        'version': '1.1.0', #当前算法版本
                        'lastVersion': '1.1.1', #最新算法版本
                        'midVersion': 1, #中间版本数
                        'runState': 1 }, #运行状态，1：运行，2：停止
               'message': 'AAA', #一般配合code使用，对异常/错误进行详细描述。
               'code': '200 OK'}  #状态码，200 OK：成功;400 Bad Request：表示客户端请求有 语法错误，不能被服务器所理解;401 Unauthonzed：表示请求未经授权;500：服务端异常。
    return jsonify(res)

#算法版模型文件请求接口
@app.route('/updateAlgorithmVersion/', methods=['POST'])
def updateAlgorithmVersion():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    else:
        res = { "message":"ABC", #一般配合code使用，对异常/错误进行详细描述。
                "code": "200 OK"}#状态码，200 OK：成功;400 Bad Request：表示客户端请求有 语法错误，不能被服务器所理解;401 Unauthonzed：表示请求未经 授权;500：服务端异常。}
    return jsonify(res)

#算法切换进度与结果请求接口
@app.route('/updateProgress/', methods=['POST'])
def updateProgress():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    else:
        res = { "message":"ABC", #一般配合code使用，对异常/错误进行详细描述。
                "code": "200 OK",#状态码，200 OK：成功;400 Bad Request：表示客户端请求有 语法错误，不能被服务器所理解;401 Unauthonzed：表示请求未经 授权;500：服务端异常。
                "progress": "100%", # 更新进度，带%
                "result": 1} # 算法更新结果：1：等待更新，2：正在更新，3：更新完成
    return jsonify(res)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=29528, threaded=False)
