"""
加载深度学习模型
"""

## 加载maskrcnn模型
from lib_inference_mrcnn import load_maskrcnn_model
maskrcnn_pointer = load_maskrcnn_model("/data/PatrolAi/maskrcnn/pointer.pth", num_classes=1, score_thresh=0.3) # 指针的maskrcnn模型

## 加载yolov5模型
from lib_inference_yolov5 import load_yolov5_model
yolov5_meter = load_yolov5_model("/data/PatrolAi/yolov5/meter.pt") # 表盘
yolov5_ErCiSheBei = load_yolov5_model("/data/PatrolAi/yolov5/ErCiSheBei.pt") ## 二次设备状态
yolov5_rec_defect_x6 = load_yolov5_model("/data/PatrolAi/yolov5/18cls_rec_defect_x6.pt") # 送检18类缺陷,x6模型
yolov5_rec_defect = load_yolov5_model("/data/PatrolAi/yolov5/18cls_rec_defect.pt") # 送检18类缺陷
# yolov5_counter= load_yolov5_model("/data/PatrolAi/yolov5/counter.pt") # 动作次数
# yolov5_digital= load_yolov5_model("/data/PatrolAi/yolov5/digital.pt") # led数字
yolov5_coco = load_yolov5_model("/data/PatrolAi/yolov5/coco.pt") # coco模型
yolov5_fire_smoke = load_yolov5_model("/data/PatrolAi/yolov5/fire_smoke.pt") # 烟火模型
yolov5_helmet = load_yolov5_model("/data/PatrolAi/yolov5/helmet.pt") # 安全帽模型
yolov5_led_color = load_yolov5_model("/data/PatrolAi/yolov5/led.pt") # led灯颜色状态模型
yolov5_ShuZiBiaoJi = load_yolov5_model("/data/PatrolAi/yolov5/ShuZiBiaoJi.pt")  # 数字表记模型
yolov5_jmjs = load_yolov5_model("/data/PatrolAi/yolov5/jmjs.pt")  # 静默监视三个设备类模型
yolov5_dz = load_yolov5_model("/data/PatrolAi/yolov5/dz.pt")  # 刀闸分析模型
# 加载ppocr模型
from lib_inference_ocr import load_ppocr
det_model_dir = "/data/PatrolAi/ppocr/ch_PP-OCRv2_det_infer/"
cls_model_dir = "/data/PatrolAi/ppocr/ch_ppocr_mobile_v2.0_cls_infer/"
rec_model_dir = "/data/PatrolAi/ppocr/ch_PP-OCRv2_rec_infer/"
text_sys = load_ppocr(det_model_dir, cls_model_dir, rec_model_dir) ## ppocr模型
