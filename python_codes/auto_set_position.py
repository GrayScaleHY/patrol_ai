import requests
import numpy as np
import logging
import time
import cv2
from common import convert_pt,get_img,convert_coor,comput_pt,comput_z,func1, func2, CurveFitting, GetInputData, get_sub_img
from lib_img_registration import registration


def get_parameters(imgs_inf):
    #data_p、data_t、data_z存rate的数组，inxex记录计算pt未出错时的索引z
    data_p, data_t, data_z, index= [], [], [1], []
    #index_z存计算z未出错时的scale,img_z存pt计算正常时的图片https
    index_z, img_z = [], []
    ret_msg = []

    for i in imgs_inf:
        print(f'i:{i}')
        z = list(i.keys())[0]
        img_path = i[list(i.keys())[0]]
        pt = i['pt']
        img_z.append(img_path[0])
        index_z.append(int(z))

        img1 = get_img(img_path[0])
        img2 = get_img(img_path[1])
        img3 = get_img(img_path[2])

        M_p= registration(img1, img2)
        M_t= registration(img1, img3)
        rate_p = comput_pt(M_p, pt, mode='p')
        rate_t = comput_pt(M_t, pt, mode='t')
        print(f'rate_p:{rate_p}')
        print(f'rate_t:{rate_t}')

        if rate_p == False or rate_t == False:
            print(f'{str(i)} 组数据，PT计算出错，请重新拍照。')
            ret_msg.append(str(i) + '组数据，PT计算出错，建议重新拍照。')
        else:
            data_p.append(rate_p)
            data_t.append(rate_t)
            index.append(int(z))
    
    print(f'index:{index}')
    print(f'data_p:{data_p}', )
    print(f'data_t:{data_t}')
    parameters_p = CurveFitting(index, data_p, 'pt')
    parameters_t = CurveFitting(index, data_t, 'pt')
    print(f'parameters_p:{parameters_p}')
    print(f'parameters_t:{parameters_t}')
    print(f'img_z:{img_z}')
    print(f'开始计算缩放倍率Z')

    for i in range(len(img_z)-1):
        print(f'i:{i}')

        img1 = get_img(img_z[i])
        img2 = get_img(img_z[i+1])

        M_z = registration(img1, img2)
        rate_z = comput_z(M_z)
        print(f'rate_z:{rate_z}')

        if rate_z > 1.0:
            data_z.append(rate_z)
        else:
            print(f'{str(i)} + 组数据Z计算出错请重新拍照。')
            ret_msg.append(str(i) + '组数据出错建议重新拍照。')
            index_z = index_z[:i+1]
            data_z = data_z[:i+1]
            break
    print(f'index_z:{index_z}')
    print(f'data_z:{data_z}')

    scale_to_z1 = [1]
    for i in range(1, len(data_z)):
        scale_to_z1.append(scale_to_z1[-1]*data_z[i])
    print(f'scale_to_z1:{scale_to_z1}')

    parameters_z = CurveFitting(index_z[0:len(data_z)], scale_to_z1, 'z')
    print(f'parameters_z:{parameters_z}')

    parameters = np.concatenate((parameters_p, parameters_t, parameters_z)).tolist()

    return parameters, ret_msg

def get_parametersV2(imgs_inf):
    index_z = list(imgs_inf[0].keys())
    img_list = list(imgs_inf[0].values())
    print(f'index_z:{index_z}')
    print(f'img_list:{img_list}')
    scale_to_pre = []

    for i in range(len(img_list)-1):
        img1 = get_img(img_list[i])
        img2 = get_img(img_list[i+1])
        M_z = registration(img1, img2)
        rate_z = comput_z(M_z)
        scale_to_pre.append(rate_z)
    print(f'index_z:{index_z}')
    print(f'scale_to_pre:{scale_to_pre}')

    scale_to_z1 = [1]
    for i in range(1, len(scale_to_pre)):
        scale_to_z1.append(scale_to_z1[-1]*scale_to_pre[i])
    print(f'scale_to_z1:{scale_to_z1}')

    parameters_z = CurveFitting(index_z[0:len(scale_to_pre)], scale_to_z1, 'z').tolist()
    print(f'parameters_z:{parameters_z}')

    out_data = {"code": 200, "parameters_z": parameters_z, "msg":"Sucess!"}

    return out_data

def calculate_ptz_coordinates(input_data):
    DATA = GetInputData(input_data)
    x, y = DATA.center # 框中心点坐标
    now_p = DATA.p
    now_t = DATA.t
    now_z = DATA.z
    range_p = DATA.range_p
    range_t = DATA.range_t
    range_z = DATA.range_z
    direction_p = DATA.direction_p
    direction_t = DATA.direction_t
    parameters = DATA.parameters
    [x1, y1, x2, y2] = DATA.rectangle_coords
    max_rate = DATA.max_rate
    resize_rate = max_rate / (max(x2 - x1, y2 - y1))

    x_rate = parameters[0] / now_z + parameters[1] #1横向单位=多少P
    y_rate = parameters[2] / now_z + parameters[3] #1纵向单位=多少T
    changeX = x - 0.5  # 需要移动的距离
    changeY = y - 0.5

    # 计算新的平移（Pan）和俯仰（Tilt）坐标，使得矩形框出现在中心
    if direction_p == 1:
        new_p = now_p + changeX*x_rate
    else:
        new_p = now_p - changeX*x_rate
    if direction_t == 1:
        new_t = now_t + changeY*y_rate
    else:
        new_t = now_t - changeY*y_rate

    new_p = convert_pt(new_p, range_p)
    new_t = convert_pt(new_t, range_t)
    print(f'new_p:{new_p},new_t:{new_t}')

    if now_z == 1:
        new_z = (resize_rate - parameters[5])/parameters[4]
    else:
        currentScale = parameters[4]*now_z + parameters[5]
        newScale = currentScale * resize_rate
        new_z = ((newScale - parameters[5])/parameters[4])
    
    if new_z < range_z[0]:
        new_z = range_z[0]
    if new_z > range_z[1]:
        new_z = range_z[1]

    data = {"ptz_new" : [new_p, new_t, new_z]}
    out_data = {"code": 200, "data": data, "msg":"Sucess!"}

    return out_data


def calculate_ptz_coordinatesV2(input_data):
    DATA = GetInputData(input_data)
    x, y = DATA.center # 框中心点坐标
    p = DATA.p
    t = DATA.t
    z = DATA.z
    fov_h = DATA.fov_h
    fov_v = DATA.fov_v
    range_p = DATA.range_p
    range_t = DATA.range_t
    range_z = DATA.range_z
    direction_p = DATA.direction_p
    direction_t = DATA.direction_t
    parameters = DATA.parameters
    parameters = parameters[-2:]
    max_rate = DATA.max_rate
    [x1, y1, x2, y2] = DATA.rectangle_coords
    resize_rate = max_rate / (max(x2 - x1, y2 - y1))
    print(f'fov_h:{fov_h}, fov_v:{fov_v}')
    # 将框子中心点坐标移到画面中心
    print(f'x:{x}, y:{y}')

    if x > 0.5: # 水平方向
        delt_x = np.rad2deg(np.arctan((x - 0.5) / 0.5 * np.tan( np.deg2rad(fov_h /2))))
    else:
        delt_x = -np.rad2deg(np.arctan((0.5 - x) / 0.5 * np.tan( np.deg2rad(fov_h /2))))
    
    if y > 0.5: # 垂直方向
        delt_y = np.rad2deg(np.arctan((y - 0.5) / 0.5 * np.tan( np.deg2rad(fov_v/ 2))))
    else:
        delt_y = -np.rad2deg(np.arctan((0.5 - y) / 0.5 * np.tan( np.deg2rad(fov_v/ 2))))
    
    print(f'delt_x:{delt_x}')
    print(f'delt_y:{delt_y}')
    if direction_p == 1:
        new_p = p + delt_x
    else:
        new_p = p - delt_x
    
    if direction_t == 1:
        new_t = t + delt_y
    else:
        new_t = t - delt_y
    print(f'new_p:{new_p},new_t:{new_t}')

    # 计算new_z
    if z == 1:
        new_z = (resize_rate - parameters[1])/parameters[0]
    else:
        currentScale = parameters[0]*z + parameters[1]
        newScale = currentScale * resize_rate
        new_z = ((newScale - parameters[1])/parameters[0])

    # if new_z < range_z[0]:
    #     new_z = range_z[0]
    # if new_z > range_z[1]:
    #     new_z = range_z[1]
    data = {"ptz_new" : [new_p, new_t, new_z]}
    out_data = {"code": 200, "data": data, "msg":"Sucess!"}

    return out_data


def calculate_ptz_coordinatesV3(input_data):
    DATA = GetInputData(input_data)
    center_x, center_y = DATA.center # 框中心点坐标
    [x1, y1, x2, y2] = DATA.rectangle_coords
    p = DATA.p
    t = DATA.t
    z = DATA.z
    fov_h = DATA.fov_h
    fov_v = DATA.fov_v
    direction_p = DATA.direction_p
    direction_t = DATA.direction_t
    parameters = DATA.parameters
    parameters = parameters[-2:]
    img1 = DATA.img1
    max_rate = DATA.max_rate
    resize_rate = max_rate / (max(x2 - x1, y2 - y1))
    requestId = DATA.requestId
    requsetUrl = DATA.requsetUrl
    h,w,_ = img1.shape

    print(f'parameters:{parameters}')
    print(f'fov_h:{fov_h}, fov_v:{fov_v}')
    print(f'ptz:{p},{t},{z}')
    print(f'center_x:{center_x}, center_y:{center_y}')
    print(f'w:{w},h:{h}')
    print(f'resize_rate1:{resize_rate}')

    #先移动一半距离
    if center_x > 0.5: # 水平方向
        delt_x1 = np.rad2deg(np.arctan((center_x - 0.5) / 0.5 * np.tan( np.deg2rad(fov_h /2)))) 
    else:
        delt_x1 = -np.rad2deg(np.arctan((0.5 - center_x) / 0.5 * np.tan( np.deg2rad(fov_h /2)))) 
    
    if center_y > 0.5: # 垂直方向
        delt_y1 = np.rad2deg(np.arctan((center_y - 0.5) / 0.5 * np.tan( np.deg2rad(fov_v/ 2)))) 
    else:
        delt_y1 = -np.rad2deg(np.arctan((0.5 - center_y) / 0.5 * np.tan( np.deg2rad(fov_v/ 2)))) 
    print(f'delt_x1:{delt_x1}')
    print(f'delt_y1:{delt_y1}')

    if direction_p == 1:
        new_p1 = p + delt_x1
    else:
        new_p1 = p - delt_x1
    
    if direction_t == 1:
        new_t1 = t + delt_y1
    else:
        new_t1 = t - delt_y1

    # 计算new_z
    if z == 1:
        new_z1 = max((0.2 / (max(x2 - x1, y2 - y1)) - parameters[1])/parameters[0], z) 
        new_z2 = (resize_rate - parameters[1])/parameters[0]
    else:
        currentScale1 = parameters[0]*z + parameters[1]
        newScale1 = currentScale1 * resize_rate * 0.2
        newScale2 = currentScale1 * resize_rate
        new_z1 = ((newScale1 - parameters[1])/parameters[0])
        new_z2 = ((newScale2 - parameters[1])/parameters[0])

    ptz1 = [new_p1, new_t1, new_z1]
    print(f'ptz1:{ptz1}')
    send_data = {'requestId' : requestId, 'ptz_coords' : ptz1}
    print(f'send_data:{send_data}')

    try:
        input_data2 = requests.post(requsetUrl, json=send_data).json()
        [current_p1, current_t1, current_z1] = input_data2['data']['ptz_current']
        FOV_H = input_data2["data"]["horizontal"]
        FOV_V = input_data2["data"]["vertical"]
        img2 = get_img(input_data2["data"]["img2"])
        # cv2.imwrite('/data/home/ckx/adjust_camera/img2_out_ori.jpg', img2)
    except:
        print(f'第二次拿图失败')
        out_data = {"code": 500, "data": None, "msg":"第二次拿图失败"}
        return out_data

    percent = 4
    while percent >= 1:
        print(f'percent:{percent}')
        sub_img = get_sub_img(img1, w, h, center_x, center_y, percent)
        M = registration(sub_img, img2)
        print(f'M:{M}')
        # cv2.imwrite('/data/home/ckx/adjust_camera/sub_img_' + str(percent) + '.jpg', sub_img)
        if M is not None and len(M) > 0:
            coor_lt = [int(x1 * w) , int(y1 * h)] #左上角的像素值
            coor_rb = [int(x2 * w) , int(y2 * h)] #右下角的像素值
            print(f'coor_lt:{coor_lt}')
            print(f'coor_rb:{coor_rb}')
            coor_tag_lt = convert_coor(coor_lt, M)
            coor_tag_rb = convert_coor(coor_rb, M)
            if all(c >= 0 for c in coor_tag_lt + coor_tag_rb):
                print(f'coor_tag_lt:{coor_tag_lt}')
                print(f'coor_tag_rb:{coor_tag_rb}')
                break  # 满足要求，跳出循环
        percent -= 1

    if M is None or len(M) == 0:
        print(f'匹配失败')
        out_data = {"code": 201, "data": None, "msg":"匹配失败"}
        return out_data

    # cv2.imwrite('/data/home/ckx/adjust_camera/img1_out_ori.jpg', img1)
    # cv2.rectangle(img1, coor_lt, coor_rb, (0, 255, 0), 2)
    # cv2.imwrite('/data/home/ckx/adjust_camera/img1_out.jpg', img1)

    # cv2.imwrite('/data/home/ckx/adjust_camera/img2_out_ori.jpg', img2)
    # cv2.rectangle(img2, coor_tag_lt, coor_tag_rb, (0, 255, 0), 2)
    # cv2.imwrite('/data/home/ckx/adjust_camera/img2_out.jpg', img2)

    center_x2, center_y2 = (coor_tag_lt[0] + coor_tag_rb[0]) /(2 * w), (coor_tag_lt[1] + coor_tag_rb[1]) / (2 * h)
    print(f'center_x2:{center_x2}, center_y2:{center_y2}')

    if center_x2 > 0.5: # 水平方向
        delt_x2 = np.rad2deg(np.arctan((center_x2 - 0.5) / 0.5 * np.tan( np.deg2rad(FOV_H /2))))
    else:
        delt_x2 = -np.rad2deg(np.arctan((0.5 - center_x2) / 0.5 * np.tan( np.deg2rad(FOV_H /2))))
    
    if center_y2 > 0.5: # 垂直方向
        delt_y2 = np.rad2deg(np.arctan((center_y2 - 0.5) / 0.5 * np.tan( np.deg2rad(FOV_V/ 2))))
    else:
        delt_y2 = -np.rad2deg(np.arctan((0.5 - center_y2) / 0.5 * np.tan( np.deg2rad(FOV_V/ 2))))
    print(f'delt_x2:{delt_x2}')
    print(f'delt_y2:{delt_y2}')

    if direction_p == 1:
        new_p2 = current_p1 + delt_x2
    else:
        new_p2 = current_p1 - delt_x2
    
    if direction_t == 1:
        new_t2 = current_t1 + delt_y2
    else:
        new_t2 = current_t1 - delt_y2
    print(f'new_p2:{new_p2},new_t2:{new_t2}')

    ptz2 = [new_p2, new_t2, new_z2]
    rec = [coor_tag_lt[0]/w , coor_tag_lt[1]/h, coor_tag_rb[0]/w, coor_tag_rb[1]/h]
    data = {"ptz_new" : ptz2, "rec" : rec}
    out_data2 = {"code": 200, "data": data, "msg":"Sucess!"}

    return out_data2


def registration_ptz_all(input_data):
    # 提取输入请求信息
    DATA = GetInputData(input_data)
    if DATA.use_fov:
        print("registration V2")
        return registration_ptzV2(input_data)
    else:
        print("registration V1")
        return registration_ptz_(input_data)
    


def registration_ptz_(input_data):
    # 提取输入请求信息
    DATA = GetInputData(input_data)
    img_ref = DATA.img_ref
    img_tag = DATA.img_tag
    parameters = DATA.parameters
    P, T, Z = DATA.p, DATA.t, DATA.z
    direction_p = DATA.direction_p
    direction_t = DATA.direction_t
    print(f'direction_p:{direction_p}')
    print(f'direction_t:{direction_t}')
    out_data = {}

    # 求img_ref上的[0.5, 0.5]点对应到img_tag上是多少。
    coor_ref = [int(img_ref.shape[1] * 0.5), int(img_ref.shape[0] * 0.5)]
    # cv2.circle(img_ref,coor_ref,3,(0,0,255), -1)
    # cv2.imwrite('/data/home/ckx/adjust_camera/ref_out.jpg', img_ref)
    print(f'coor_ref:{coor_ref}')
    M = registration(img_ref, img_tag)
    print(f'M:{M}')

    if M is None:
        out_data["code"] = 500
        out_data["ptz_new"] = None
        out_data["msg"] = "Error"
        return out_data
    
    coor_tag = convert_coor(coor_ref, M)
    print(f'coor_tag:{coor_tag}')
    # cv2.circle(img_tag,coor_tag,3,(0,0,255), -1)
    # cv2.imwrite('/data/home/ckx/adjust_camera/tag_out.jpg', img_tag)
    # dx、dy为tag图上的点距离中心点的距离, x_rate, y_rate
    dx = coor_tag[0] / img_tag.shape[1] - 0.5
    dy = coor_tag[1] / img_tag.shape[0] - 0.5
    print(f'dx:{dx}')
    print(f'dy:{dy}')
    k_x_rate = parameters[0]; b_x_rate = parameters[1]
    k_y_rate = parameters[2]; b_y_rate = parameters[3]

    x_rate = (k_x_rate / Z) + b_x_rate
    y_rate = (k_y_rate / Z) + b_y_rate
    print(f'x_rate:{x_rate}, x_rate * dx:{x_rate * dx}')
    print(f'y_rate:{y_rate}, y_rate * dy:{y_rate * dy}')
    ## 求变化后的PTZ，dP / dx = x_rate
    tag_P = True
    tag_T = True

    if abs(dx) > 0.01:
        if int(direction_p) == 1:
            P_raw = P + x_rate * dx
        else:
            P_raw = P - x_rate * dx
    else:
        tag_P = False
        P_raw = P    
    if abs(dy) > 0:
        if int(direction_t) == 1:
            T_raw = T + y_rate * dy
        else:
            T_raw = T - y_rate * dy
    else:
        tag_T =False
        T_raw = T

    if tag_P == False and tag_T ==False:
        out_data["code"] = 201
    else:
        out_data["code"] = 200

    data = {"ptz_new" : [P_raw, T_raw, Z], "offset_angle" : [abs(dx * x_rate), abs(dy * y_rate)], "offset_percent" : [abs(dx), abs(dy)]}    
    out_data["data"] = data
    out_data["msg"] = "Success!"

    return out_data


def registration_ptzV2(input_data):
    DATA = GetInputData(input_data)
    img_ref = DATA.img_ref
    img_tag = DATA.img_tag
    p = DATA.p
    t = DATA.t
    z = DATA.z
    fov_h = DATA.fov_h  # 水平
    fov_v = DATA.fov_v  # 垂直
    direction_p = DATA.direction_p
    direction_t = DATA.direction_t

    print(f'fov_h:{fov_h}, fov_v:{fov_v}')

    out_data = {}
    # 求img_ref上的[0.5, 0.5]点对应到img_tag上是多少。
    H, W = img_ref.shape[:2]
    coor_ref = [int(W * 0.5), int(H * 0.5)]

    # use osd
    cv2.rectangle(img_ref, (0, 0), (int(W*4/10), int(H/20)), (0, 0, 0), thickness=-1)
    cv2.rectangle(img_ref, (0, int(H*14/15)), (int(W/6), H), (0, 0, 0), thickness=-1)
    M = registration(img_ref, img_tag)
    # M = lightglue_registration(img_ref, img_tag)

    if M is None:
        out_data["code"] = 500
        out_data["ptz_new"] = None
        out_data["msg"] = "Error"
        return out_data
    
    coor_tag = convert_coor(coor_ref, M)
    # x、y为tag图上的点距离中心点的距离
    x = coor_tag[0] / img_tag.shape[1]
    y = coor_tag[1] / img_tag.shape[0]

    if x > 0.5: # 水平方向
        delt_x = np.rad2deg(np.arctan((x - 0.5) / 0.5 * np.tan( np.deg2rad(fov_h /2))))
    else:
        delt_x = -np.rad2deg(np.arctan((0.5 - x) / 0.5 * np.tan( np.deg2rad(fov_h /2))))
    
    if y > 0.5: # 垂直方向
        delt_y = np.rad2deg(np.arctan((y - 0.5) / 0.5 * np.tan( np.deg2rad(fov_v/ 2))))
    else:
        delt_y = -np.rad2deg(np.arctan((0.5 - y) / 0.5 * np.tan( np.deg2rad(fov_v/ 2))))
    
    print(f'delt_x:{delt_x}')  # 相对于图片中心点的x偏转角度
    print(f'delt_y:{delt_y}')
    if direction_p == 1:
        new_p = p + delt_x
    else:
        new_p = p - delt_x
    
    if direction_t == 1:
        new_t = t + delt_y
    else:
        new_t = t - delt_y
    print(f'new_p:{new_p},new_t:{new_t}')


    data = {"ptz_new" : [new_p, new_t, z * 2], "offset_angle" : [delt_x, delt_y], "offset_percent" : [abs(x), abs(y)]}
    out_data = {"code": 200, "data": data, "msg":"Sucess!"}

    return out_data





if __name__ == '__main__':
    # 设置日志记录
    logging.basicConfig(
        level=logging.DEBUG,  # 设置日志级别为 DEBUG
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/data/home/ckx/adjust_camera/logfile.log'),  # 指定日志保存的文件路径
            logging.StreamHandler()  # 同时也可以将日志输出到控制台
        ]
    )
    
    # 测试配置工具3.0
    get_url = "http://192.168.44.217:31010/api/v1/channel/getGisInfo" 
    get_pic = "http://192.168.44.217:31010/api/v1/channel/capture"
    data = {"chnid": 5430}

    cam_info = requests.get(get_url, params=data).json()
    print(f'cam_info:{cam_info}')
    P = cam_info["data"]["PanPos"]
    T = cam_info["data"]["TiltPos"]
    Z = cam_info["data"]["ZoomPos"]
    FOV_H = cam_info["data"]["Horizontal"]
    FOV_V = cam_info["data"]["Vertical"]
    img1 = get_img(requests.get(get_pic, params=data).json()["data"]["picData"])
    # 0.005122950819672131, 0.9497307001795332, 0.020491803278688523, 0.992818671454219
    # 0.005122950819672131, 0.003590664272890485, 0.022540983606557378, 0.0466786355475763
    # 0.9743852459016393, 0.005385996409335727, 0.992827868852459, 0.04488330341113106
    # 0.9733606557377049, 0.947935368043088, 0.9907786885245902, 0.9838420107719928
    # 0.045081967213114756, 0.7809694793536804, 0.09221311475409837, 0.822262118491921
    # 0.5778688524590164, 0.599640933572711, 0.6096311475409836, 0.6804308797127468
    input_data =  {'rectangle_coords': [0.5778688524590164, 0.599640933572711, 0.6096311475409836, 0.6804308797127468], 
                   'ptz_coords': [P, T, Z], 
                   'Horizontal': FOV_H, 
                   'Vertical': FOV_V, 
                   'direction_p': 1, 
                   'direction_t': 1, 
                   'range_p': [0, 360], 
                   'range_t': [355, 90], 
                   'range_z': [1, 25], 
                   'parameters': [54.44637174168189, 1.5899789710686194, 31.536435849580215, 0.8287580477135175, 0.7788835705412782],
                   'img1' : img1,
                   'max_rate' : 0.8}
    out_data = calculate_ptz_coordinatesV3(input_data)

    ## 纠偏测试
    # ref = '/data/home/ckx/adjust_camera/ref.jpg'
    # tag = '/data/home/ckx/adjust_camera/tag.jpg'
    # input_data = {
    # "img_ref": ref,  
    # "img_tag": tag, 
    # "ptz_coords": [267.3999938964844, 1.600000023841858, 92.0],
    # "parameters": [54.44637174168189, 1.5899789710686194, 31.536435849580215, 0.8287580477135175, 0.7788835705412782],
    # "direction_p": 2,  
    # "direction_t": 2, 
    # }
    # registration_ptz_(input_data)

