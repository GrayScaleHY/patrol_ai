from lib_img_registration import registration
from lib_help_base import GetInputData

import sys
import torch


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

    out_data = {"code": 0, "msg": "Request; ",
                "checkpoint": checkpoint}

    if an_type != "position_shift":
        out_data["code"] = 1
        out_data["msg"] = out_data["msg"] + "Type is wrong! "
        return out_data

    if img_ref is None:
        out_data["code"] = 1
        out_data["msg"] = out_data["msg"] + "Reference image is None! "
        return out_data

    if img_tag is None:
        out_data["code"] = 1
        out_data["msg"] = out_data["msg"] + "Target image is None! "
        return out_data

    h, w, _ = img_ref.shape
    M = registration(img_ref, img_tag)

    if M is None:
        out_data["code"] = 1
        out_data["msg"] = out_data["msg"] + "Preset offset! "
        return out_data

    print('shift matrix:{}'.format(M.tolist()))
    result = analyse_shift_matrix(M.tolist(), h, w)
    if result:
        out_data["code"] = 1
        out_data["msg"] = out_data["msg"] + "Preset offset! "
    else:
        out_data["msg"] = out_data["msg"] + "Normal preset position. "
    return out_data


if __name__ == '__main__':
    from lib_image_ops import img2base64
    import cv2
    img0 = cv2.imread('/data/NBPatrolAi/result_patrol/ref.png')
    img0 = img2base64(img0)
    img1 = cv2.imread('/data/NBPatrolAi/result_patrol/ref.png')
    h, w, c = img1.shape
    img1 = cv2.resize(img1[:int(h*0.7), :int(w*0.7), :], (w, h))
    img1 = img2base64(img1)
    input_data = {"type": "position_shift", "image": img1,
                  "checkpoint": "0",
                  "config": {"img_ref": img0}}
    print(check_position(input_data))
