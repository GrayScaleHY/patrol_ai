"""
巡检算法平台测试，缺陷识别测试代码。
python util_qxsb.py --source <images dir> --out <result path>
"""

import os
import cv2
from lib_inference_yolov5 import load_yolov5_model, inference_yolov5
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source',
    type=str,
    default='./qxsb',
    help='source dir.')
parser.add_argument(
    '--out_dir',
    type=str,
    default='./result/40zhytdlkjgfyxgs',
    help='out file of saved result.')
parser.add_argument(
    '--conf',
    type=float,
    default=0.25,
    help='threshold of confidence.')
parser.add_argument(
    '--iou',
    type=float,
    default=0.55,
    help='threshold of iou.')
parser.add_argument(
    '--model_dir',
    type=str,
    default='/data/PatrolAi/yolov5',
    help='model path.')
parser.add_argument(
    '--data_part',
    type=str,
    default='1/1',
    help='part of data split.')
args, unparsed = parser.parse_known_args()

in_dir = args.source # 待测试文件目录
out_dir = args.out_dir # 输出结果文件
conf_thr = args.conf # confidence 阈值
iou_thr = args.iou # iou 阈值
model_dir = args.model_dir ## 模型存放的目录
data_part = args.data_part # 分隔数据部分

## 加载模型
weights = os.path.join(model_dir, "18cls_rec_defect_x6.pt")
yolov5_weights = load_yolov5_model(weights)

## 创建文件夹
os.makedirs(out_dir, exist_ok=True)

## 分割数据
img_list = os.listdir(in_dir)
img_list.sort()
_s = int(data_part.split("/")[1])
_p = int(data_part.split("/")[0])
_l = len(img_list)
if _s != _p:
    img_list = img_list[int(_l*(_p-1)/_s):int(_l*_p/_s)]
else:
    img_list = img_list[int(_l*(_p-1)/_s):]

## 批处理
for img_name in img_list:
    out_file = os.path.join(out_dir, img_name[:-4] + ".txt")
    loop_start = time.time()

    img_file = os.path.join(in_dir, img_name) # 读取图片
    img = cv2.imread(img_file)

    ## 模型推理
    bbox_cfg = inference_yolov5(yolov5_weights, img, resize=1280, conf_thres=conf_thr, iou_thres=iou_thr) #推理
    print(img_file)
    print(bbox_cfg)

    count = 1
    f = open(out_file, "w", encoding='utf-8')
    f.write("ID,PATH,TYPE,SCORE,XMIN,YMIN,XMAX,YMAX\n")
    ## 保存推理结果
    for bbox in bbox_cfg:
        label = bbox["label"]
        c = bbox["coor"]

        ID = str(count)
        PATH = img_name
        TYPE = bbox["label"]
        SCORE = "1.0000"
        XMIN = str(c[0]); YMIN = str(c[1]); XMAX = str(c[2]); YMAX = str(c[3])

        ## 输出结果
        result = [ID,PATH,TYPE,SCORE,XMIN,YMIN,XMAX,YMAX]
        f.write(",".join(result) + "\n")
        count += 1
    f.close()
    print(f"loop time = {time.time() - loop_start}")



