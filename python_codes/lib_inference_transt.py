
## 追踪算法推理代码
# https://github.com/dschoerk/TransT

import sys
sys.path.insert(0,'../TransT') ## ultralytics/yolov5 存放的路径
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from pysot_toolkit.bbox import get_axis_aligned_bbox
from pysot_toolkit.trackers.tracker import Tracker
from pysot_toolkit.trackers.net_wrappers import NetWithBackbone

def load_transt(net_path):
    """
    加载模型
    """
    try:
        net = NetWithBackbone(net_path=net_path, use_gpu=True)
        print("Load transt model with gpu !")
    except:
        net = NetWithBackbone(net_path=net_path, use_gpu=False)
        print("load transt model with cpu ! ")
    tracker = Tracker(name='transt', net=net, window_penalty=0.49, exemplar_size=128, instance_size=256)
    return tracker

def init_tracker(tracker, img, bbox):
    """
    初始化模型，输入初始帧以及目标框。
    """
    gt_bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
    init_info = {'init_bbox': gt_bbox}
    tracker.initialize(img, init_info)
    return tracker

def track(tracker, img):
    """
    追踪模型推理
    return:
        tracker: 重新初始化的追踪器
        bbox: 追踪后的目标框
    """
    outputs = tracker.track(img)
    pre = outputs['target_bbox']
    bbox = [int(pre[0]), int(pre[1]), int(pre[0]+pre[2]), int(pre[1]+pre[3])]
    return tracker, bbox

def track_bboxes(tracker, img_ref, img_tag, gt_bboxes):
    pre_bboxes = []
    for gt_bbox in gt_bboxes:
        tracker = init_tracker(tracker, img_ref, gt_bbox)
        tracker, pre_bbox = track(tracker, img_tag)
        pre_bboxes.append(pre_bbox)
    return tracker, pre_bboxes

if __name__ == '__main__':
    import time
    import cv2
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1" 


    net_path = "/home/yh/task/model/transt.pth"
    ref = "tag.jpg"
    tag = "ref.jpg"
    
    img_ref = cv2.imread(ref)
    img_tag = cv2.imread(tag)
    bboxes = [[542, 417, 815, 682], 
              [1040, 390, 1316, 652], 
              [574, 746, 609, 781],
              [1037, 866, 1075, 897]]

    # bboxes = [[736, 128, 801, 199], 
    #           [589, 923, 668, 1022], 
    #           [884, 628, 1009, 758],
    #           [1096, 144, 1178, 220]]
    
    for bbox in bboxes:
        cv2.rectangle(img_ref, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    cv2.imwrite("ref_result.jpg", img_ref)

    start = time.time()
    tracker = load_transt(net_path)
    print("load model time:", time.time() - start)

    tracker, pre_bboxes = track_bboxes(tracker, img_ref, img_tag, bboxes)

    start = time.time()
    tracker, pre_bboxes = track_bboxes(tracker, img_ref, img_tag, bboxes)
    print("tracker time:", time.time() - start)

    for bbox in pre_bboxes:
        cv2.rectangle(img_tag, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

    cv2.imwrite("tag_result.jpg", img_tag)

