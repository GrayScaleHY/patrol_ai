from lib_inference_yolov5 import load_yolov5_model, inference_yolov5
from lib_inference_yolov5 import load_yolov5seg_model, inference_yolov5seg
from lib_inference_yolov8 import load_yolov8_model, inference_yolov8, inference_yolov8seg

def load_yolo_model(model_file, decode=False):
    # Load a model
    try:
        model = load_yolov5_model(model_file)
    except:
        try:
            model = load_yolov5seg_model(model_file)
        except:
            model = load_yolov8_model(model_file)
    return model


def inference_yolo(model_yolo, img, resize=640, conf_thres=0.2, iou_thres=0.2, pre_labels=None):
    try:
        res = inference_yolov5(model_yolo, img, resize=resize, conf_thres=conf_thres, iou_thres=iou_thres, pre_labels=pre_labels)
    except:
        try:
            res = inference_yolov5seg(model_yolo, img, resize=resize, conf_thres=conf_thres, iou_thres=iou_thres, pre_labels=pre_labels)
        except:
            try:
                res = inference_yolov8(model_yolo, img, resize=resize, conf_thres=conf_thres, iou_thres=iou_thres, pre_labels=pre_labels)
            except:
                res = inference_yolov8seg(model_yolo, img, resize=resize, conf_thres=conf_thres, iou_thres=iou_thres, pre_labels=pre_labels)

    return res