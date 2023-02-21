from lib_inference_yolov5 import load_yolov5_model, inference_yolov5
from lib_inference_yolov5 import load_yolov5seg_model, inference_yolov5seg
from lib_inference_yolov8 import load_yolov8_model, inference_yolov8


def load_yolo_model(model_file, decode=False):
    # Load a model
    try:
        model = load_yolov5_model(model_file)
    except Exception as e:
        try:
            model = load_yolov5seg_model(model_file)
        except Exception as e:
            model = load_yolov8_model(model_file)
    return model


def inference_yolo(model_yolo, img, resize=640, conf_thres=0.2,same_iou_thres=0.7,
                     diff_iou_thres=1,focus_labels=None):
    try:
        res = inference_yolov5(model_yolo, img, resize=resize, conf_thres=conf_thres, iou_thres=same_iou_thres,
                               pre_labels=focus_labels)
    except Exception as e:
        try:
            res = inference_yolov5seg(model_yolo, img, resize=resize, conf_thres=conf_thres, iou_thres=same_iou_thres,
                                      pre_labels=focus_labels)
        except Exception as e:
                res = inference_yolov8(model_yolo, img, resize=resize, conf_thres=conf_thres, same_iou_thres=same_iou_thres,
                                       diff_iou_thres=diff_iou_thres,focus_labels=focus_labels)

    return res
