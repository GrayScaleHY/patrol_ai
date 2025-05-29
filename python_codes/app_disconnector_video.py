
from lib_help_base import color_area, GetInputData, creat_img_result, reg_crop
from util_yjsk import video_states, final_state
from lib_image_ops import img2base64, img_chinese
import glob
import time
import json
import cv2
import numpy as np
import os


def creat_img_result(input_data, img_tag_):
    """
    巡视算法输出的img_result兼容图片路径和base64.
    """
    if "image" in input_data and  os.path.exists(input_data["image"]): 
        out_file = input_data["image"][:-4] + "_result.jpg"
        cv2.imwrite(out_file, img_tag_)
        img_result = out_file
    else:
        img_result = img2base64(img_tag_)
    return img_result
    
def inspection_disconnector_video(input_data):
    """
    刀闸识别
    """
    ## 提取输入请求信息
    DATA = GetInputData(input_data)
    checkpoint = DATA.checkpoint
    video_path = DATA.video_path
    an_type = DATA.type

    map = {
        1: "合闸正常",
        2: "分闸正常",
        3: "合闸异常",
        4: "分闸异常",
    }
    states_start, states_end, img_tag = video_states(video_path) # 求tag_video的状态列表
    f_state = final_state(states_start, states_end, len_window=4) # 求最终状态
    
    out_data = {"code": 0, "data": {"result": map[f_state]}, "img_result":"image"}
    ## 画上点位名称
    img_tag_ = img_chinese(img_tag, checkpoint + an_type , (10, 10), color=(255, 0, 0), size=60)
    img_tag_ = img_chinese(img_tag_, map[f_state] , (500, 100), color=(0, 0, 255), size=60)
    out_data["img_result"] = creat_img_result(input_data, img_tag_)  # 返回结果图
    return out_data

if __name__ == '__main__':
    from lib_help_base import get_save_head, save_input_data, save_output_data
    input_data = {
        "video_path": "/data/PatrolAi/test_images/yjsk/合闸异常.mp4",
        "type": "disconnector_video"
    }
    start = time.time()
    out_data = inspection_disconnector_video(input_data)
    print("spend time:", time.time() - start)
    for s in out_data:
        if s != "img_result":
            print(s,":",out_data[s])
    
    save_dir, name_head = get_save_head(input_data)
    save_input_data(input_data, save_dir, name_head, draw_img=True)
    save_output_data(out_data, save_dir, name_head)

