import requests
import json
import os
import cv2
from lib_inference_yolov5 import load_yolov5_model, inference_yolov5

yolov5_red_defect= load_yolov5_model("/data/inspection/yolov5/meter.pt") # 表盘

def get_contest_info():
    url = "http://10.85.XXX.XXX:XXX/contest/get_contest_info"
    contest_info = requests.get(url=url).json()
    print(contest_info)


def post_submit_result(result_data):
    url = "http://10.85.XXX.XXX:XXX/contest/submit_result"
    result_text = json.dumps(result_data)
    res = requests.post(url=url, data=result_text).json()
    print(res)


if __name__ == '__main__':
    img_dir = "tuilishuju"
    list_file = os.path.join(img_dir, "train_list.txt")
    contestantId = "49e6264e517d428796b532796b74a364"

    
    count = 1
    result = []
    for line in open(list_file, "r", encoding='utf-8'):
        img_file = line.strip()
        if not os.path.exists(img_file):
            print("Warning:",img_file,"not exist !")
            continue

        img = cv2.imread(img_file)
        if img is None:
            print("Warning:", img_file, "name is not rule !")
        bbox_cfg = inference_yolov5(yolov5_red_defect, img, resize=640, conf_thres=0.4, iou_thres=0.4)

        for cfg in bbox_cfg:
            id_ = str(count)
            path = img_file
            type_ = cfg["label"]
            score = str(cfg["score"])
            xmin = str(cfg["coor"][0])
            ymin = str(cfg["coor"][1])
            xmax = str(cfg["coor"][2])
            ymax = str(cfg["coor"][3])

            result.append({"id": id_, "path": path, "type": type_, "score": score, "xmin":xmin, "ymin": ymin, "xmax":xmax, "ymax":ymax})
            count += 1
    
    result_data = {"contestantId":contestantId,"isEnd":1,"results":result}
    post_submit_result(result_data)


