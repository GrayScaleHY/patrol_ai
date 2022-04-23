
from lib_image_ops import base642img, img2base64, img_chinese
from util_inspection_disconnector import disconnector_state
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

def get_input_data(input_data):
    """
    提取input_data中的信息。
    return:
        img_tag: 目标图片数据
        img_open: 模板图，刀闸打开
        img_close: 模板图，刀闸闭合
        roi1, roi2: 感兴趣区域, 结构为[xmin, ymin, xmax, ymax]
    """

    img_tag = base642img(input_data["image"])

    ## 是否有模板图
    img_close = None
    if "img_close" in input_data["config"]:
        if isinstance(input_data["config"]["img_close"], str):
            img_close = base642img(input_data["config"]["img_close"])    

    ## 是否有模板图
    img_open = None
    if "img_open" in input_data["config"]:
        if isinstance(input_data["config"]["img_open"], str):
            img_open = base642img(input_data["config"]["img_open"]) 
        
    ## 感兴趣区域
    roi1= None # 初始假设
    roi2= None # 初始假设
    if "bboxes" in input_data["config"]:
        if isinstance(input_data["config"]["bboxes"], dict):
            if "roi1" in input_data["config"]["bboxes"]:
                if isinstance(input_data["config"]["bboxes"]["roi1"], list):
                    if len(input_data["config"]["bboxes"]["roi1"]) == 4:
                        W = img_open.shape[1]; H = img_open.shape[0]
                        roi1 = input_data["config"]["bboxes"]["roi1"]
                        roi1 = [int(roi1[0]*W), int(roi1[1]*H), int(roi1[2]*W), int(roi1[3]*H)]
            if "roi1" in input_data["config"]["bboxes"]:
                if isinstance(input_data["config"]["bboxes"]["roi2"], list):
                    if len(input_data["config"]["bboxes"]["roi2"]) == 4:
                        W = img_open.shape[1]; H = img_open.shape[0]
                        roi2 = input_data["config"]["bboxes"]["roi2"]
                        roi2 = [int(roi2[0]*W), int(roi2[1]*H), int(roi2[2]*W), int(roi2[3]*H)]
    
    return img_tag, img_open, img_close, roi1, roi2

def inspection_disconnector(input_data):
    """
    刀闸识别
    """
    TIME_START = time.strftime("%m-%d-%H-%M-%S")
    save_dir = os.path.join(os.path.join("inspection_result/disconnector",TIME_START)) #保存图片的路径
    os.makedirs(save_dir, exist_ok=True)
    f = open(os.path.join(save_dir, "input_data.json"), "w")
    json.dump(input_data, f, ensure_ascii=False)  # 保存输入信息json文件
    f.close()
    
    ## 提取data信息
    out_data = {"code": 0, "data":{}, "msg": "Success request disconnector"}
    img_tag, img_open, img_close, roi1, roi2 = get_input_data(input_data)

    ## 保存模板图与待分析图
    cv2.imwrite(os.path.join(save_dir,"img_close.jpg"), img_close)
    cv2.imwrite(os.path.join(save_dir,"img_tag.jpg"), img_tag)
    cv2.imwrite(os.path.join(save_dir,"img_open.jpg"), img_open)

    ## 求刀闸状态
    bboxes = [roi1, roi2]
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
    tag_file = "/home/yh/image/python_codes/test/test1/img_tag2.jpg"
    open_file = "/home/yh/image/python_codes/test/test1/img_open.jpg"
    close_file = "/home/yh/image/python_codes/test/test1/img_close.jpg"
    
    bboxes = [[651, 315, 706, 374], [661, 400, 713, 450]]
    img_close = img2base64(cv2.imread(close_file))
    img_open = img2base64(cv2.imread(open_file))
    img_tag = img2base64(cv2.imread(tag_file))
    H, W = cv2.imread(close_file).shape[:2]
    roi1 = [bboxes[0][0] / W, bboxes[0][1] / H, bboxes[0][2] / W, bboxes[0][3] / H]
    roi2 =  [bboxes[1][0] / W, bboxes[1][1] / H, bboxes[1][2] / W, bboxes[1][3] / H]
    input_data = {
        "image": img_tag,
        "config": {"img_open": img_open, "img_close": img_close, "bboxes": {"roi1": roi1, "roi2": roi2}},
        "type": "disconnector"
    }
    out_data = inspection_disconnector(input_data)
    for s in out_data:
        if s != "img_result":
            print(s,":",out_data[s])

