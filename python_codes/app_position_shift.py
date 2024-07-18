from lib_img_registration import registration, convert_coor, correct_offset
from lib_help_base import GetInputData, creat_img_result


def analyse_shift_matrix(shift_matrix, h, w):
    """
    预置位偏移矩阵分析
    """
    # 阈值越小越宽松
    w_thres = 0.8
    # 阈值越大越宽松
    bias_thres = [w * 0.3, h * 0.3]
    thres = [[w_thres, w_thres, bias_thres[i]] for i in range(2)]
    result = False
    for index in range(6):
        i, j = index // 3, index % 3
        if index in [0, 4]:
            if abs(shift_matrix[i][j]) < abs(thres[i][j]) or abs(shift_matrix[i][j]) > 2-abs(thres[i][j]):
                print('超:{}'.format(index))
                result = True
        elif index in [1, 3]:
            if abs(shift_matrix[i][j]) > 1 - abs(thres[i][j]):
                print('超:{}'.format(index))
                result = True
        else:
            if abs(shift_matrix[i][j]) > abs(thres[i][j]):
                print('超:{}'.format(index))
                result = True
        if result:
            break
    return result


def check_position(input_data):
    """
    检查预置位是否偏移
    """
    # 提取输入请求信息
    DATA = GetInputData(input_data)
    checkpoint = DATA.checkpoint
    an_type = DATA.type
    img_ref = DATA.img_ref
    img_tag = DATA.img_tag

    out_data = {"code": 1, "msg": "Request; ",
                "checkpoint": checkpoint}

    if an_type != "position_shift":
        out_data["msg"] = out_data["msg"] + "Type is wrong! "
        return out_data

    if img_ref is None:
        out_data["msg"] = out_data["msg"] + "Reference image is None! "
        return out_data

    if img_tag is None:
        out_data["msg"] = out_data["msg"] + "Target image is None! "
        return out_data

    h, w, _ = img_ref.shape

    # 输出初始化
    out_data = {"code": 1,
                "data": {"no_roi": [{"label": "1", "bbox": [0, 0, int(w), int(h)]}]},
                "img_result": creat_img_result(input_data, img_tag),
                "msg": "Request; ",
                "checkpoint": checkpoint}


    M = registration(img_ref, img_tag)

    if M is None:
        out_data["msg"] = out_data["msg"] + "Preset offset! "
        return out_data

    print('shift matrix:{}'.format(M.tolist()))
    result = analyse_shift_matrix(M.tolist(), h, w)
    if result:
        out_data["msg"] = out_data["msg"] + "Preset offset! "
    else:
        out_data["code"] = 0
        out_data["data"]["no_roi"][0]["label"] = "0"
        out_data["data"]["no_roi"][0]["bbox"] = []
        out_data["msg"] = out_data["msg"] + "Normal preset position. "
    return out_data

def registration_ptz(input_data):
    # 提取输入请求信息
    DATA = GetInputData(input_data)
    checkpoint = DATA.checkpoint
    an_type = DATA.type
    img_ref = DATA.img_ref
    img_tag = DATA.img_tag
    par = input_data["config"]["parameters"]
    P, T, Z = input_data["ptz_coords"]

    out_data = {"code": 1, "data": {"raw_ptz": []}, "img_result": "image", "msg": "Request;"}

    # 求img_tag上的[0.5, 0.5]点对应到img_ref上是多少。
    coor_tag = [int(img_tag.shape[1] * 0.5), int(img_tag.shape[0] * 0.5)]
    M = registration(img_tag, img_ref)
    img_ref_warped = correct_offset(img_ref, M)
    out_data["img_result"] = creat_img_result(input_data, img_ref_warped) # 返回结果图
    if M is None:
        return out_data
    
    coor_ref = convert_coor(coor_tag, M)

    # 计算dx, dy, x_rate, y_rate
    dx = 0.5 - coor_ref[0] / img_ref.shape[1]
    dy = 0.5 - coor_ref[1] / img_ref.shape[0]

    k_x_rate = par[0]; b_x_rate = par[1]
    k_y_rate = par[2]; b_y_rate = par[3]

    x_rate = (k_x_rate / Z) + b_x_rate
    y_rate = (k_y_rate / Z) + b_y_rate

    ## 求变化后的PTZ，dP / dx = x_rate
    P_raw = x_rate * dx + P
    if P_raw > 360:
        P_raw = P_raw - 360
    if P_raw < 0:
        P_raw = 360 + P_raw

    T_raw = y_rate * dy + T
    if T_raw > 360:
        T_raw = T_raw - 360
    if T_raw < 0:
        T_raw = 360 + T_raw

    out_data["code"] = 0
    out_data["data"] = {"raw_ptz": [P_raw, T_raw, Z]}

    return out_data

if __name__ == '__main__':
    from lib_help_base import get_save_head, save_input_data, save_output_data


    # from lib_image_ops import img2base64
    # import cv2
    # img0 = cv2.imread('/data/NBPatrolAi/result_patrol/ref.png')
    # img0 = img2base64(img0)
    # img1 = cv2.imread('/data/NBPatrolAi/result_patrol/ref.png')
    # h, w, c = img1.shape
    # img1 = cv2.resize(img1[:int(h*0.7), :int(w*0.7), :], (w, h))
    # img1 = img2base64(img1)
    # input_data = {"type": "position_shift", "image": img1,
    #               "checkpoint": "0",
    #               "config": {"img_ref": img0}}
    # print(check_position(input_data))

    # P1, T1, Z1 = 316.3999938964844, 1.7999999523162842, 4.699999809265137

    # P, T, Z = 313.1000061035156, 359.70001220703125, 4.699999809265137
    input_data = {
        "checkpoint": "巡视点位01",
        "image":"images/img_tag.jpg",
        "ptz_coords": [313.1000061035156, 359.70001220703125, 4.699999809265137],
        "config":{
            "img_ref": "images/img_ref.jpg",
            "parameters": [54.12504, 2.047665, 30.513250, 0.774448, 0.758836, 0.345861]
        },
        "type": "registration"
    }

    out_data = registration_ptz(input_data)
    save_dir, name_head = get_save_head(input_data)
    save_input_data(input_data, save_dir, name_head, draw_img=True)
    save_output_data(out_data, save_dir, name_head)
    print("------------------------------")
    for s in out_data:
        if s != "img_result":
            print(s, ":", out_data[s])
    print("-------------------------------")
