"""
加载深度学习模型
"""

## 加载maskrcnn模型
from lib_inference_mrcnn import load_maskrcnn_model
maskrcnn_pointer = load_maskrcnn_model("/data/inspection/maskrcnn/pointer.pth", num_classes=1, score_thresh=0.3) # 加载指针的maskrcnn模型

from lib_inference_yolov5 import load_yolov5_model
yolov5_meter = load_yolov5_model("/data/inspection/yolov5/meter.pt") # 表盘
yolov5_ErCiSheBei = load_yolov5_model("/data/inspection/yolov5/ErCiSheBei.pt") ## 二次设备状态
yolov5_rec_defect = load_yolov5_model("/data/inspection/yolov5/rec_defect_x6.pt") # 北京送检17类缺陷
yolov5_counter= load_yolov5_model("/data/inspection/yolov5/counter.pt") # 动作次数
yolov5_digital= load_yolov5_model("/data/inspection/yolov5/digital.pt") # led数字
yolov5_coco = load_yolov5_model("/data/inspection/yolov5/coco.pt") # coco模型

from lib_inference_ocr import load_ppocr
det_model_dir = "/data/inspection/ppocr/ch_PP-OCRv2_det_infer/"
cls_model_dir = "/data/inspection/ppocr/ch_ppocr_mobile_v2.0_cls_infer/"
rec_model_dir = "/data/inspection/ppocr/ch_PP-OCRv2_rec_infer/"
text_sys = load_ppocr(det_model_dir, cls_model_dir, rec_model_dir) ## 加载ppocr模型