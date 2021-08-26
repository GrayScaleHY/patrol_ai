
import math
import numpy as np
import cv2
import os
import json
from lib_image_ops import base642img

def segment2angle(base_coor, tar_coor):
    """
    输入线段的两个端点坐标（图像坐标系，y轴朝下），返回该线段斜率转换为0-360度。
    args:
        base_coor: 基本坐标点，(x1, y1)
        tar_coor: 目标点,(x2, y2)
    return:
        返回正常直角坐标系的角度，0-360度。
    """
    dx = tar_coor[0] - base_coor[0]
    dy = tar_coor[1] - base_coor[1]
    if dx == 0:
        if dy > 0:
            angle = 270
        else:
            angle = 90
    else:
        tan = dy / dx
        if tan > 0:
            if dx > 0:
                angle = 360 - (math.atan(tan) * 180 / math.pi)
            else:
                angle = 180 - (math.atan(tan) * 180 / math.pi)
        else:
            if dx > 0:
                angle = -(math.atan(tan) * 180 / math.pi)
            else:
                angle = 180 - (math.atan(tan) * 180 / math.pi)
    return angle


def angle_scale(config):
    """
    根据刻度与角度的关系，求出相差一度对应的刻度差。
    args:
        config: list, 角度与刻度的对应关系。格式如下
                [{"136.39":0.0, "90.26":4.5, "72.12":8.0}, ..]
    return:
        out_cfg: list, 最小刻度和角度，单位刻度。格式如下：
                [[141.7, -0.1, 0.018]]
    """
    out_cfg = []
    for scale_config in config:
        assert len(scale_config) >= 2, "刻度信息至少要有两个。"

        # 将config转换成浮点数字型，并置于array中。
        config_list = np.array([[0, 0]]*len(scale_config), dtype=float)
        count = 0
        for ang in scale_config:
            config_list[count][0] = float(ang)
            config_list[count][1] = float(scale_config[ang])
            count += 1

        # 找出最小刻度的行index
        min_index = np.where(config_list[:, 1] == min(config_list[:, 1]))[0][0]
        util_scale = 0
        for i, cfg in enumerate(config_list):
            if i != min_index:

                # 根据指针大刻度在顺时针方向的原则获取角度跨度。
                if cfg[0] < config_list[min_index][0]:
                    util_scale += (cfg[1] - config_list[min_index]
                                   [1]) / (config_list[min_index][0] - cfg[0])
                else:
                    util_scale += (cfg[1] - config_list[min_index][1]) / \
                        (360 - cfg[0] + config_list[min_index][0])

        util_scale = util_scale / (len(config_list) - 1)
        out_cfg.append([config_list[min_index][0],
                       config_list[min_index][1], util_scale])
        # print(str(scale_config) + " --> " +
        #       str([config_list[min_index][0], config_list[min_index][1], util_scale]))

    return out_cfg


def angle2sclae(cfg, ang):
    """
    根据角度计算刻度。
    args:
        cfg: 刻度盘属性：[最小刻度对应的角度， 最小刻度值， 单位刻度]
        ang: 需要计算的角度
    return:
        scale: 指针的刻度。
    """
    if ang < cfg[0]:
        scale = cfg[1] + (cfg[0] - ang) * cfg[2]
    else:
        scale = cfg[1] + (cfg[0] + 360 - ang) * cfg[2]
    return scale


def draw_result(input_data, out_data, save_img):
    """
    根据out_data画图。
    """
    save_path = os.path.dirname(save_img)
    os.makedirs(save_path, exist_ok=True)

    f = open(os.path.join(save_path, "input_data.json"), "w", encoding='utf-8')
    json.dump(input_data, f, ensure_ascii=False, indent=2, sort_keys=True)
    f.close()

    f = open(os.path.join(save_path, "out_data.json"), "w", encoding='utf-8')
    json.dump(out_data, f, ensure_ascii=False, indent=2, sort_keys=True)
    f.close()

    raw_img = os.path.join(save_path, "raw_img.jpg")
    img = base642img(input_data["image"])
    cv2.imwrite(raw_img, img)

    for cfg in out_data["data"]:
        
        if cfg["type"] == "counter":
            for i, bbox_coor in enumerate(cfg["bboxes"]):
                bbox_coor = [int(c) for c in bbox_coor]
                cv2.rectangle(img, (int(bbox_coor[0]), int(bbox_coor[1])),
                      (int(bbox_coor[2]), int(bbox_coor[3])), (0, 0, 255), thickness=2)
                cv2.putText(img, str(cfg["values"][i]), (int(bbox_coor[0])-5, int(bbox_coor[1])-5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), thickness=2)
                    
        elif cfg["type"] == "pointer":
            for i, segment in enumerate(cfg['segments']):
                segment = [int(c) for c in segment]
                cv2.line(img, (segment[0], segment[1]), (segment[2], segment[3]), (0, 255, 0), 2)
                cv2.putText(img, "%.3f"%(cfg['values'][i]), (segment[2]-5, segment[3]-5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), thickness=2)

            meter_coor = [int(c) for c in cfg["bbox"]]
            cv2.rectangle(img, (meter_coor[0], meter_coor[1]),
                      (meter_coor[2], meter_coor[3]), (0, 0, 255), thickness=2)
            cv2.putText(img, 'meter', (meter_coor[0]-5, meter_coor[1]-5),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), thickness=2)
        
        elif cfg["type"] == "meter":
            for i, bbox_coor in enumerate(cfg["bboxes"]):
                bbox_coor = [int(c) for c in bbox_coor]
                cv2.rectangle(img, (int(bbox_coor[0]), int(bbox_coor[1])),
                      (int(bbox_coor[2]), int(bbox_coor[3])), (0, 0, 255), thickness=2)
                cv2.putText(img, "meter", (int(bbox_coor[0])-5, int(bbox_coor[1])-5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), thickness=2)

    cv2.imwrite(save_img, img)
