
from lib_image_ops import base642img, img2base64, img_chinese
from util_inspection_disconnector import disconnector_state, json2bboxes
import glob
import time
import json
import cv2
import numpy as np
import os

state_map = {
    "合": {"name": "合闸正常", "color": [(255,0,0), (255,0,0)]},
    "分": {"name": "分闸正常", "color": [(0,255,0), (0,255,0)]},
    "异常": {"name": "分合闸异常", "color": [(0,0,255), (0,0,255)]},
    "无法判别状态": {"name": "分析失败", "color": [(0,0,255), (0,0,255)]},
}

def inspection_disconnector_video(input_data):
    """
    刀闸识别
    """
    cfg_dir = "/export/patrolservice/VIDEO"

    TIME_START = time.strftime("%m-%d-%H-%M-%S")
    save_dir = os.path.join(os.path.join("inspection_result/disconnector_video",TIME_START)) #保存图片的路径
    os.makedirs(save_dir, exist_ok=True)
    f = open(os.path.join(save_dir, "input_data.json"), "w")
    json.dump(input_data, f, ensure_ascii=False)  # 保存输入信息json文件
    f.close()
    
    ## 提取data信息
    out_data = {"code": 0, "data":{}, "msg": "Success request disconnector"}
    video_path = input_data["video_path"]

    ## 截取视频最后一帧
    cap = cv2.VideoCapture(video_path) ## 建立视频对象
    while(cap.isOpened()):
        ret, frame = cap.read() # 逐帧读取
        if frame is not None:
            img_tag = frame
        if not ret:
            break
    
    ## 求刀闸状态
    json_file = glob.glob(os.path.join(cfg_dir, "*.json"))[0]
    open_file = glob.glob(os.path.join(cfg_dir, "*_off.png"))[0]
    close_file = glob.glob(os.path.join(cfg_dir, "*_on.png"))[0]
    assert os.path.exists(json_file) and os.path.exists(open_file) and os.path.exists(close_file), "模板文件不全"
    img_open = cv2.imread(open_file)
    img_close = cv2.imread(close_file)
    bboxes = json2bboxes(json_file, img_open)
    state, bboxes_tag = disconnector_state(img_open, img_close, img_tag, bboxes)

    out_data["data"] = {"result": state_map[state]["name"]}

    ## 保存结果
    for i in range(2):
        b1 = bboxes[0]; b2 = bboxes[1]
        bt1 = bboxes_tag[0]; bt2 = bboxes_tag[1]
        cv2.rectangle(img_open, (b1[0], b1[1]), (b1[2], b1[3]), state_map["分"]["color"][0], thickness=2)
        cv2.rectangle(img_open, (b2[0], b2[1]), (b2[2], b2[3]), state_map["分"]["color"][0], thickness=2)
        cv2.rectangle(img_close, (b1[0], b1[1]), (b1[2], b1[3]), state_map["合"]["color"][0], thickness=2)
        cv2.rectangle(img_close, (b2[0], b2[1]), (b2[2], b2[3]), state_map["合"]["color"][0], thickness=2)
        cv2.rectangle(img_tag, (bt1[0], bt1[1]), (bt1[2], bt1[3]), state_map[state]["color"][0], thickness=2)
        cv2.rectangle(img_tag, (bt2[0], bt2[1]), (bt2[2], bt2[3]), state_map[state]["color"][1], thickness=2)
    img_tag = img_chinese(img_tag, state_map[state]["name"], (10, 50), color=state_map[state]["color"][0], size=40)

    cv2.imwrite(os.path.join(save_dir,"img_open_cfg.jpg"), img_open)
    cv2.imwrite(os.path.join(save_dir,"img_close_cfg.jpg"), img_close)
    cv2.imwrite(os.path.join(save_dir,"img_tag_cfg.jpg"), img_tag)

    f = open(os.path.join(save_dir,"output_data.json"),"w",encoding='utf-8')
    json.dump(out_data, f, indent=2)
    f.close()

    out_data["img_result"] = img2base64(img_tag)

    return out_data

if __name__ == '__main__':
    input_data = {
        "video_path": "/data/inspection/disconnector_video/192.168.1.64-8000-37-1652944807847.mp4",
        "type": "disconnector_video"
    }
    start = time.time()
    out_data = inspection_disconnector_video(input_data)
    print("spend time:", time.time() - start)
    for s in out_data:
        if s != "img_result":
            print(s,":",out_data[s])

