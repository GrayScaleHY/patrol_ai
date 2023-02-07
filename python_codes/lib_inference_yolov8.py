from ultralytics import YOLO

def load_yolov8_model(model_file, decode=False):
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO(model_file)  # load a pretrained model (recommended for training)
    return model


def inference_yolov8(model_yolov8, img, resize=640, conf_thres=0.2, iou_thres=0.2, pre_labels=None):
    result = model_yolov8(img)
    labels = model_yolov8.module.names if hasattr(model_yolov8, 'module') else model_yolov8.names
    bbox_cfg = []
    for cfg in result:
        res = cfg.boxes
        for box in res:
            cfg = {'coor': list(list(box.xyxy.cpu().numpy())[0]), 'score': float(box.conf),
                   'label': labels[int(box.cls)]}
            bbox_cfg.append(cfg)
            print(cfg)
    return bbox_cfg


def load_yolov8seg_model(model_file='/data/PatrolAi/yolov5/daozha_seg.pt', decode=False):
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO(model_file)  # load a pretrained model (recommended for training)
    return model


def inference_yolov8seg(model_yolov8, img, resize=640, conf_thres=0.2, iou_thres=0.2, pre_labels=None):
    result = model_yolov8(img)
    return


if __name__ == '__main__':
    import shutil
    import cv2

    img_file = "1009013581.jpg"
    weight = "yolov8n.pt"
    img = cv2.imread(img_file)
    model_yolov5 = load_yolov8_model(weight)

    cfgs = inference_yolov8(model_yolov5, img, resize=640, conf_thres=0.7, iou_thres=0.5)
    print('cfgs', cfgs)
    for cfg in cfgs:
        # res=cfg.boxes
        # for box in res:
        # cfg={'coor':list(list(box.xyxy.cpu().numpy())[0]),'score':float(box.conf),'label':int(box.cls)}
        # print(cfg)
        c = cfg["coor"]
        label = cfg["label"]
        score = cfg["score"]
        cv2.rectangle(img, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (255, 0, 255), thickness=2)
        cv2.putText(img, str(label) + ": " + str(score), (int(c[0]), int(c[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 255), thickness=2)
    cv2.imwrite(img_file[:-4] + "result.jpg", img)
    print(cfgs)