from ultralytics import YOLO

def load_yolov8_model(model_file,decode=False):
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO(model_file)  # load a pretrained model (recommended for training)
    return model

def inference_yolov8(model_yolov8, img, resize=640, conf_thres=0.2, iou_thres=0.2, pre_labels=None):
    result = model_yolov8(img)
    return

def load_yolov8seg_model(model_file='/data/PatrolAi/yolov5/daozha_seg.pt',decode=False):
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO(model_file)  # load a pretrained model (recommended for training)
    return model

def inference_yolov8seg(model_yolov8, img, resize=640, conf_thres=0.2, iou_thres=0.2, pre_labels=None):
    result = model_yolov8(img)
    return