from flask import Flask, request, jsonify
import json
from lib_help_base import get_save_head, save_input_data, save_output_data, rm_result_patrolai
import os
from app_pointer import inspection_pointer
# from app_disconnector import inspection_disconnector
# from app_counter import inspection_counter
from app_object_detection import inspection_object_detection
from app_qrcode_ocr import inspection_qrcode
from app_panbie import inspection_identify_defect
from config_version import code_version
# from app_disconnector_video import inspection_disconnector_video
from app_yeweiji import inspection_level_gauge
# from app_ocr import ocr_digit_detection
from app_yejingpingshuzishibie import inspection_digital_rec
from app_match import patrol_match
from app_position_shift import check_position, registration_ptz
import time
import threading
from config_object_name import AI_FUNCTION, convert_ai_function
from app_led_video import patrolai_led_video
from app_jmjs import patrolai_jmjs
from config_object_name import jmjs_dict
from lib_help_base import GetInputData, traverse_and_modify
from app_adjust_camera import adjust_camera
from app_shuzishibie_video import inspection_digital_rec_video
from app_sticker import inspection_sticker_detection

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from auto_set_position import get_parameters, get_parametersV2, calculate_ptz_coordinates, calculate_ptz_coordinatesV2, \
    calculate_ptz_coordinatesV3, registration_ptz_all

## 单独开个进程，定时删除result_patrolai文件夹中的文件。
t = threading.Thread(target=rm_result_patrolai,args=())
t.start()

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

draw_img = True

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False # 让jsonify返回的json串支持中文
app.url_map.strict_slashes = False

## ai能力列表
@app.route('/AI_function/', methods=['GET'])
def inspection_ai_function():
    return jsonify(convert_ai_function(AI_FUNCTION))

@app.route('/adjust_camera_v2/', methods=['POST'])
def adjust_camera_v2_server():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))
    
    start_time = time.time()
    save_dir, name_head = get_save_head(data)
    save_input_data(data, save_dir, name_head, draw_img=False)
    res = adjust_camera(data)
    f = open(os.path.join(save_dir, name_head + "output_data.json"), "w", encoding='utf-8')
    json.dump(res, f, ensure_ascii=False)  # 保存输入信息json文件
    f.close()

    print(name_head, "spend time:", round(time.time() - start_time, 3), "out data:", traverse_and_modify(res))

    return jsonify(res)

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

    print(name_head, "spend time:", round(time.time() - start_time, 3), "out data:", traverse_and_modify(res))

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

    print(name_head, "spend time:", round(time.time() - start_time, 3), "out data:", traverse_and_modify(res))

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

    print(name_head, "spend time:", round(time.time() - start_time, 3), "out data:", traverse_and_modify(res))

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

    print(name_head, "spend time:", round(time.time() - start_time, 3), "out data:", traverse_and_modify(res))

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

    print(name_head, "spend time:", round(time.time() - start_time, 3), "out data:", traverse_and_modify(res))

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
@app.route('/inspection_person/', methods=['POST'])
@app.route('/inspection_biaoshipai/', methods=['POST'])
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

    print(name_head, "spend time:", round(time.time() - start_time, 3), "out data:", traverse_and_modify(res))

    return jsonify(res)

@app.route('/inspection_rec_defect/', methods=['POST']) # 识别缺陷
def inspection_rec_server():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))
    
    start_time = time.time()
    save_dir, name_head = get_save_head(data)
    save_input_data(data, save_dir, name_head, draw_img=draw_img)

    DATA = GetInputData(data)
    label_list = DATA.label_list
    jmjs_list = [i for i in jmjs_dict]
    if len(label_list) > 0 and len(list(set(label_list) & set(jmjs_list))) == len(label_list):
        res = patrolai_jmjs(data)
    else:
        res = inspection_object_detection(data)

    save_output_data(res, save_dir, name_head)

    print(name_head, "spend time:", round(time.time() - start_time, 3), "out data:", traverse_and_modify(res))

    return jsonify(res)

# 判别算法
@app.route('/inspection_identify_defect/', methods=['POST'])
def inspection_identify_defect_server():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))
    
    start_time = time.time()
    save_dir, name_head = get_save_head(data)
    save_input_data(data, save_dir, name_head, draw_img=draw_img)
    res = inspection_identify_defect(data)
    save_output_data(res, save_dir, name_head)
    
    print(name_head, "spend time:", round(time.time() - start_time, 3), "out data:", traverse_and_modify(res))

    return jsonify(res)

# 预置位偏移
@app.route('/inspection_position_shift/', methods=['POST'])
def inspection_position_server():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))

    start_time = time.time()
    # 根据input_data和当前时刻获取保存文件夹和文件名头
    save_dir, name_head = get_save_head(data)
    save_input_data(data, save_dir, name_head, draw_img=draw_img)
    res = check_position(data)
    save_output_data(res, save_dir, name_head)
    
    print(name_head, "spend time:", round(time.time() - start_time, 3), "out data:", traverse_and_modify(res))

    return jsonify(res)

# 预置位偏移
@app.route('/inspection_registration/', methods=['POST'])
def inspection_registration_server():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))

    start_time = time.time()
    # 根据input_data和当前时刻获取保存文件夹和文件名头
    save_dir, name_head = get_save_head(data)
    save_input_data(data, save_dir, name_head, draw_img=draw_img)
    res = registration_ptz(data)
    save_output_data(res, save_dir, name_head)
    
    print(name_head, "spend time:", round(time.time() - start_time, 3), "out data:", traverse_and_modify(res))
    
    return jsonify(res)


@app.route('/inspection_shuzi_video/', methods=['POST'])
def inspection_shuzi_video_server():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))

    start_time = time.time()
    # 根据input_data和当前时刻获取保存文件夹和文件名头
    save_dir, name_head = get_save_head(data)
    save_input_data(data, save_dir, name_head, draw_img=draw_img)
    res = inspection_digital_rec_video(data)
    save_output_data(res, save_dir, name_head)

    print(name_head, "spend time:", round(time.time() - start_time, 3), "out data:", traverse_and_modify(res))

    return jsonify(res)

@app.route('/inspection_sticker/', methods=['POST'])
def inspection_sticker_server():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))

    start_time = time.time()
    # 根据input_data和当前时刻获取保存文件夹和文件名头
    save_dir, name_head = get_save_head(data)
    save_input_data(data, save_dir, name_head, draw_img=draw_img)
    res = inspection_sticker_detection(data)
    save_output_data(res, save_dir, name_head)

    print(name_head, "spend time:", round(time.time() - start_time, 3), "out data:", traverse_and_modify(res))

    return jsonify(res)


# ## 刀闸分合识别
# @app.route('/inspection_disconnector/', methods=['POST'])
# def inspection_disconnector_server():
#     if request.method != 'POST':
#         res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
#         return jsonify(res)
#     data = json.loads(request.get_data(as_text=True))
#     res = inspection_disconnector(data)
#     print("inspection_pointer result:")
#     print("-----------------------------------------------")
#     for s in res:
#         if s != "img_result":
#             print(s,":",res[s])
#     print("----------------------------------------------")
#     return jsonify(res)

# ## 视频分析刀闸分合状态
# @app.route('/inspection_disconnector_video/', methods=['POST'])
# def inspection_disconnector_video_server():
#     if request.method != 'POST':
#         res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
#         return jsonify(res)
#     data = json.loads(request.get_data(as_text=True))
#     res = inspection_disconnector_video(data)
#     print("inspection_pointer result:")
#     print("-----------------------------------------------")
#     for s in res:
#         if s != "img_result":
#             print(s,":",res[s])
#     print("----------------------------------------------")
#     return jsonify(res)

## led灯视频分析
@app.route('/inspection_led_video/', methods=['POST'])
def patrolai_led_video_server():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    data = json.loads(request.get_data(as_text=True))
    res = patrolai_led_video(data)
    print("patrolai_led_video result:")
    print("-----------------------------------------------")
    for s in res:
        if s != "img_result":
            print(s, ":", res[s])
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

@app.route('/get_ptz_parameters/', methods=['POST'])
def get_ptz_parameters():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    t1 = time.time()
    data = json.loads(request.get_data(as_text=True))
    print(f'data: {data}')

    res, ret_msg = get_parameters(data['imgs_inf'])
    print(f'res: {res}')
    print(f'ret_msg: {ret_msg}')
    print(f'cost time: {time.time()-t1}')

    return jsonify(res, ret_msg)

@app.route('/adjust_cameraV2/', methods=['POST'])
def adjust_cameraV2():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    
    t1 = time.time()
    data = json.loads(request.get_data(as_text=True))

    print(f'data:{data}')
    res = calculate_ptz_coordinatesV2(data)
    print(f'res:{res}')
    print(f'cost time:{time.time()-t1}')

    return jsonify(res)


@app.route('/adjust_cameraV3/', methods=['POST'])
def adjust_cameraV3():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    
    t1 = time.time()
    data = json.loads(request.get_data(as_text=True))
    # print(f'data:{data}')

    res = calculate_ptz_coordinatesV3(data)
    print(f'res:{res}')
    print(f'cost time:{time.time()-t1}')
    
    return jsonify(res)

@app.route('/registration_ptz/', methods=['POST'])
def registration_ptz():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    
    data = json.loads(request.get_data(as_text=True))
    t1 = time.time()
    print(f'data:{data}')

    res = registration_ptz_all(data)
    print(f'res:{res}')
    print(f'cost time:{time.time()-t1}')

    return jsonify(res)

#算法版本获取接口
@app.route('/Version/', methods=['POST'])
def Version():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    else:
        # 返回的数据
        res = {
                'Version': "1.1.040715_release", #版本ID
                'lastVersion':"1.1.040715_release", #z最新版本
                'midVersion': 0, #中间版本数
                }
    return jsonify(res)

#模型更新接口
@app.route('/updateModel/', methods=['POST'])
def updateModel():
    if request.method != 'POST':
        res = {'code': 1, 'msg': 'Only POST requests are supported!', 'data': []}
        return jsonify(res)
    else:
        res = {
                "result": 3} # 算法更新结果：1：等待更新，2：正在更新，3：更新完成
    return jsonify(res)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=29528, threaded=False)
