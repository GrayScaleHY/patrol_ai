import numpy as np
import cv2
import os
import json
import time
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image 
from ais_bench.infer.interface import InferSession
from lib_rcnn_ops import filter_cfgs

## pip install ultralytics==8.0.32
from ultralytics.yolo.utils.ops import xywh2xyxy, process_mask_native, scale_boxes, crop_mask, clip_boxes

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm = 0,  # number of masks
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Arguments:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_boxes, num_classes + 4 + num_masks)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nm (int): The number of masks output by the model.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    
    ### mod
    prediction = torch.tensor(prediction)
    device = 'cpu'
    prediction.to(device)
    ###

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[1] - nm - 4  # number of classes
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x.transpose(0, -1)[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)
        box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        if multi_label:
            i, j = (cls > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING ?? NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def postprocess(preds,
                re_size=(640, 640), 
                raw_size=(640, 640),
                conf_thres=0.5,
                iou_thres=0.8):
    """
    args:
        preds: session.infer的结果
        re_size: 送入网络的图片大小(h, w)
        raw_size: 图片原始大小(H, W)
    return:
        results: ["bboxes": ]
    """
    if len(preds) == 2: ## 分割模型
        nm = 32; multi_label = True
    else: # 检测模型
        nm = 0; multi_label = False
    p = non_max_suppression(prediction=preds[0],
                            conf_thres=conf_thres,
                            iou_thres=iou_thres,
                            multi_label=multi_label,
                            max_det=100,
                            nm=nm)
    results = []
    for i, pred in enumerate(p):
      
        if not len(pred):
            return []
        
        else:
            pred[:, :4] = scale_boxes(re_size, pred[:, :4], raw_size).round()

            if len(preds) == 2: # 分割模型
                proto = preds[1]
                masks = process_mask_native(torch.from_numpy(proto[i]), pred[:, 6:], pred[:, :4], raw_size)  # HWC
            else:
                masks = [None for i in range(len(pred))]
        
        results.append({"boxes":pred[:, :6], "masks":masks})
    
    return results

def load_yolov8_model(model_file):
    """
    加载模型
    """
    device_id = 0
    session = InferSession(device_id, model_file)
    if model_file.endswith(".pt"):
        model_file = model_file[:-3] + ".om"
    label_file = model_file[:-3] + ".json"

    if os.path.exists(label_file):
        f = open(label_file, "r", encoding='utf-8')
        label_dict = json.load(f)
        f.close()
        labels = label_dict["labels"]
    else:
        # labels = [str(i) for i in range(100)]
        labels = ["pointer"] + [str(i + 1) for i in range(100)]

    yolov8_weights = [session, labels]
    return yolov8_weights

def inference_yolov8(yolov8_weights, 
                     img, 
                     resize=640, 
                     conf_thres=0.2, 
                     same_iou_thres=0.7, 
                     diff_iou_thres=1, 
                     focus_labels=None):
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

    ## 前处理
    raw_size = img.shape[:2]
    re_size = (resize, resize)
    img, _, _ = letterbox(img, new_shape=re_size)
    img = np.expand_dims(img, axis=0)
    img = img[..., ::-1].transpose(0, 3, 1, 2)  # BGR tp RGB
    image_np = np.array(img, dtype=np.float32)
    image_np_expanded = image_np / 255.0
    img = np.ascontiguousarray(image_np_expanded).astype(np.float32)

    # 模型推理
    session, labels = yolov8_weights
    outputs = session.infer([img])

    #后处理
    preds = postprocess(outputs, re_size,raw_size, conf_thres=conf_thres,iou_thres=same_iou_thres)
    
    if len(preds) == 0:
        return []
    
    preds = preds[0] # 只有一张图
    
    cfgs = []
    for i in range(len(preds['boxes'])):
        
        box = preds['boxes'][i].tolist()
        mask = preds['masks'][i]
        
        cfg={'label': labels[int(box[5])],
            'coor':[int(x) for x in box[:4]],
            'score': round(box[4], 5),
            'mask':mask}
        
        cfgs.append(cfg)  
    
    # 根据conf_thres、iou_thres、focus_labels过滤结果
    cfgs = filter_cfgs(cfgs, conf_thres, same_iou_thres, diff_iou_thres, focus_labels=focus_labels)
        
    return cfgs

def postprocess_classify(preds):
        results = []
        for i, pred in enumerate(preds):
            results.append({"probs":pred})
        return results

def inference_yolov8_classify(yolov8_weights, img, resize=640):
    
    img = Image.fromarray(img)
    
    tfl = T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR)
    img = tfl(img)
    tfl = T.CenterCrop(resize)
    img = tfl(img)
    
    DEFAULT_MEAN = (0.0, 0.0, 0.0)
    DEFAULT_STD = (1.0, 1.0, 1.0)
    tr = T.Compose([T.ToTensor(),T.Normalize(torch.tensor(DEFAULT_MEAN),torch.tensor(DEFAULT_STD))])
    image_np_expanded = tr(img)
    
    img = torch.stack([image_np_expanded],dim=0)
    img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).float()

    # 模型推理
    session, labels = yolov8_weights
    outputs = session.infer([img])

    #后处理
    preds = postprocess_classify(outputs)
    if len(preds) == 0:
        return []
    
    cfgs = []
    #print(preds)
    
    preds = preds[0]['probs'][0]
    top5i = torch.tensor(preds).argsort(0, descending=True)[:5].tolist()
    #print(top5i[0])
    class_name = labels[top5i[0]]#top5i[0]#
    prob = preds[top5i[0]]
        
    cfgs.append({'label':class_name,'score':prob, "coor": [], "mask":[]})  
        
    return cfgs

if __name__ == '__main__':

    import time
    img_file = "/data/PatrolAi/result_patrol/he.jpg"
    label_file = ""
    model_file = '/data/PatrolAi/result_patrol/yjsk.om' 
    img = cv2.imread(img_file)
    yolov8_weights = load_yolov8_model(model_file, label_file)
    cfgs = inference_yolov8_classify(yolov8_weights,img)
    print(cfgs)
