"""
巡检算法平台测试，缺陷识别测试代码。
python util_inspection_rec_defect.py --source <images dir> --out <result path>
"""

import os
import cv2
from lib_inference_yolov5 import load_yolov5_model, inference_yolov5
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source',
    type=str,
    default='./qxsb',
    help='source dir.')
parser.add_argument(
    '--out',
    type=str,
    default='./40zhytdlkjgfyxgs.txt',
    help='out file of saved result.')
parser.add_argument(
    '--conf',
    type=float,
    default=0.25,
    help='threshold of confidence.')
parser.add_argument(
    '--iou',
    type=float,
    default=0.2,
    help='threshold of iou.')
parser.add_argument(
    '--model_dir',
    type=str,
    default='/data/inspection/yolov5',
    help='model path.')
args, unparsed = parser.parse_known_args()

in_dir = args.source # 待测试文件目录
out_file = args.out # 输出结果文件
conf_thr = args.conf # confidence 阈值
iou_thr = args.iou # iou 阈值
model_dir = args.model_dir ## 模型存放的目录

## 加载模型
weights = os.path.join(model_dir, "rec_defect_x6.pt")
yolov5_weights = load_yolov5_model(weights)

## 创建文件夹
os.makedirs(os.path.dirname(out_file), exist_ok=True)

## 批处理
count = 1
f = open(out_file, "w", encoding='utf-8')
f.write("ID,PATH,TYPE,SCORE,XMIN,YMIN,XMAX,YMAX\n")
for img_name in os.listdir(in_dir):

    img_file = os.path.join(in_dir, img_name) # 读取图片
    img = cv2.imread(img_file)

    ## 模型推理
    bbox_cfg = inference_yolov5(yolov5_weights, img, resize=1280, conf_thres=conf_thr, iou_thres=iou_thr) #推理
    print(img_file)
    print(bbox_cfg)

    ## 保存推理结果
    for bbox in bbox_cfg:
        label = bbox["label"]
        c = bbox["coor"]

        ID = str(count)
        PATH = img_name
        TYPE = bbox["label"]
        SCORE = "1.0"
        XMIN = str(c[0]); YMIN = str(c[1]); XMAX = str(c[2]); YMAX = str(c[3])

        ## 输出结果
        result = [ID,PATH,TYPE,SCORE,XMIN,YMIN,XMAX,YMAX]
        f.write(",".join(result) + "\n")
        count += 1

f.close()

