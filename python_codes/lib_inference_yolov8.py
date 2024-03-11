"""
yolov8的分类、检测、分割推理代码。
https://docs.ultralytics.com/python/
"""

from ultralytics import YOLO
import ultralytics
import numpy as np
from lib_rcnn_ops import filter_cfgs
import cv2
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_yolov8_model(model_file, decode=False):
    """
    加载yolov8模型。
    """
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    # load a pretrained model (recommended for training)
    model = YOLO(model_file)
    # model.to(device)
    return model


def inference_yolov8(model,
                     img,
                     resize=None,
                     conf_thres=0.25,
                     same_iou_thres=0.7,
                     diff_iou_thres=1,
                     focus_labels=None
                     ):
    """
    使用yolov5对图片做推理，返回推理结果。
    args:
        model: 加载后的yolov8模型，使用load_yolov8_model(model_file)加载。
        img: 待分析图片数据，numpy图片数据。
        resize: 图片resize大小。
        conf_thres: 置信度阈值。
        same_iou_thres: 同类标签之间的iou阈值。
        diff_iou_thres: 所有目标物之间的iou阈值。
        focus_labels: 关注的标签。若focus_labels=None,则不过滤；若focus_labels为list，则过滤focus_labels以外的标签。
    return:
        cfgs: 预测结果，格式为[{"label": "", "coor": [x0, y0, x1, y1], "score": float, "mask": numpy}, {}, ..]
    """
    labels = model.names  # 标签名
    img_ = img.copy()
    
    if resize is None:
        resize = model.model.args["imgsz"]

    result = model(img_, iou=same_iou_thres, conf=conf_thres, imgsz=resize)[0]  # 推理结果
    task = model.task  # 模型类型, detect, segment, classify
    H, W = img.shape[:2]

    assert task in ["detect", "segment", "classify"], "is not yolov8 model !"

    # 分类模型推理结果后处理
    if task == "classify":
        res = result.probs.data.cpu().numpy()
        index = np.argmax(res)
        score = float(res[index])
        label = str(labels[index])
        if score < conf_thres:
            cfgs = []
        else:
            cfgs = [{"label": label, "coor": [], "score": score, "mask":[]}]
        return cfgs

    # 检测和分割模型推理结果后处理
    res_boxes = result.boxes.cpu().numpy()
    if str(ultralytics.__version__) == "8.0.51":
        boxes = res_boxes.boxes
    else:
        boxes = res_boxes.data
    if len(boxes) == 0:
        return[]
    
    if task == "segment":
        res_masks = result.masks.cpu().numpy()
        res_segments = result.masks.segments

    cfgs = []
    img_shape = np.array([W, H])
    for i, lab in enumerate(res_boxes.cls):
        b = list(res_boxes.xyxy[i])  # 目标框坐标
        box = [int(b[0]), int(b[1]), int(b[2]), int(b[3])]
        label = labels[lab]  # 标签
        score = res_boxes.conf[i]  # 得分

        if task == "segment":
            if str(ultralytics.__version__) == "8.0.51":
                mask = res_masks.masks[i].astype(np.uint8)  # 分割模型的mask
            else:
                mask = res_masks.data[i].astype(np.uint8)  # 分割模型的mask
            mask = cv2.resize(mask, (W, H)) # 将mask映射为原始图片大小
            segments = res_segments[i] * img_shape
            segments = segments.astype(int)
            if np.sum(segments) == 0: # 没有mask的情况
                continue
        else:
            mask = None
            segments = None

        cfgs.append({"label": label, "coor": box,
                    "score": score, "mask": mask, "segments": segments})

    if len(cfgs) == 0:
        return []
        
    # 根据conf_thres、iou_thres、focus_labels过滤结果
    cfgs = filter_cfgs(cfgs, conf_thres, same_iou_thres,
                       diff_iou_thres, focus_labels)

    return cfgs


if __name__ == '__main__':
    import shutil
    import cv2
    import time

    img_file = "/data/PatrolAi/result_patrol/pointer/0615135355_2号主变2号低抗线温表-视频_tag.jpg"
    weight = "/data/PatrolAi/yolov8/pointer.pt"
    img = cv2.imread(img_file)
    img_raw = img.copy()
    print("input image shape:", img.shape)

    start = time.time()
    model_yolov5 = load_yolov8_model(weight)
    print("load model spend time:", time.time() - start)

    start = time.time()
    cfgs = inference_yolov8(model_yolov5, img, resize=640, conf_thres=0.25,
                            same_iou_thres=0.8, diff_iou_thres=1, focus_labels=None)
    print("inference image spend time:", time.time() - start)

    for i in range(len(cfgs)):
        tmp = cfgs[i]
        # print(tmp)
        c = tmp["coor"]
        label = tmp["label"]
        score = str(tmp["score"]).zfill(3)
        # mask = tmp['mask']
        # print(mask)
        # cv2.polylines(img, [mask], isClosed=True, color=(0, 0, 255), thickness=1)
        cv2.rectangle(img_raw, (int(c[0]), int(c[1])),
                      (int(c[2]), int(c[3])), (0, 255, 0), 2)
        cv2.putText(img_raw, label + ": " + score, (int(c[0]), int(
            c[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)

    cv2.imwrite('bus_result.jpg', img_raw)
