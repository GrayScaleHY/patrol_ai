# -*- coding: utf-8 -*-
# import selectivesearch # pip install selectivesearch
import cv2
import numpy as np


def selective_search(img_file):
    """
    选择性搜索方法计算感兴趣区域
    paper: Selective Search for Object Recognition
    讲解: https://cloud.tencent.com/developer/article/1614718
    return:
        candidates: 感兴趣区域框, 结构[[x, y, w, h], ...]
    """
    img = cv2.imread(img_file) # loading image

    ## perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)

    ## 筛选区域框
    candidates = set()
    for r in regions:
        x, y, w, h = r['rect']

        ## 剔除面积小于图像0.005的区域，删除长宽比大于1.5的区域。
        if r['size'] <= img.size / 100 or w / h > 1.5 or h / w > 1.5:
            continue

        candidates.add(r['rect'])
    candidates = list(candidates)

    ## draw rectangles on the original image
    # for x, y, w, h in candidates:
    #     cv2.rectangle(img, (x, y),(x+w, y+h), (0, 0, 255), thickness=2)
    # cv2.imwrite("result_selective_search.jpg",img)

    return candidates


def iou(vertice1, vertice2):
    """
    这是计算两个矩形区域的交并比函数
    args:
        vertice1, vertice2: 目标框, 格式为:[xin,ymin,xmax,ymax]
    return: 
        两个矩形区域的交并比
    """
    vertice1 = np.array(vertice1, dtype=float)
    vertice2 = np.array(vertice2, dtype=float)

    # 计算区域交集的左上与右下坐标
    lu = np.maximum(vertice1[0:2], vertice2[0:2])
    rd = np.minimum(vertice1[2:], vertice2[2:])

    # 计算区域交集的面积
    intersection = np.maximum(0.0, rd - lu)
    inter_square = intersection[0] * intersection[1]

    # 计算区域并集的面积
    square1 = (vertice1[2] - vertice1[0]) * (vertice1[3] - vertice1[1])
    square2 = (vertice2[2] - vertice2[0]) * (vertice2[3] - vertice2[1])
    union_square = np.maximum(square1 + square2 - inter_square, 1e-10)

    return np.clip(inter_square / union_square, 0.0, 1.0)


def nms(dets, thresh):
    """
    NMS, 非极大抑制，消除多余重叠比例较高的目标框。
    https://cloud.tencent.com/developer/article/1614720
    args:
        dets: 目标框数组，目标框的格式为：[xmin,ymin,xmax,ymax,score]
        thresh: 阈值
    return: 
        keep: 不重复的目标框数组在元目标框数组中的下标数组
    """

    vertices = dets[:, 0:4]  # 目标框
    scores = dets[:, 4]  # 分值

    #areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 打分从大到小排列，取index
    order = scores.argsort()[::-1]
    # keep为最后保留的边框
    keep = []
    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)
        # 计算窗口i与其他所有窗口的交叠部分的面积
        ious = np.array([iou(vertices[i], vertices[j]) for j in order[1:]])
        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ious <= thresh)[0]
        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]
    return keep


def filter_cfgs(cfgs, conf_thres=0.25, same_iou_thres=1, diff_iou_thres=1, focus_labels=None):
    """
    根据conf_thres、iou_thres和pre_labels过滤推理结果。
    args:
        cfgs: 模型推理结果，结构为 [{"label": "", "coor": [x0, y0, x1, y1], "score": float, "mask": mask}, {}, ..]
        conf_thres: 置信度阈值。
        same_iou_thres: 同类标签之间的iou阈值。
        diff_iou_thres: 所有目标物之间的iou阈值。
        focus_labels: 关注的标签。若focus_labels=None,则不过滤；若focus_labels为list，则过滤focus_labels以外的标签。
    return:
        cfgs: 过滤后的模型推理结果,格式和输入的cfgs一样。
    """
    ## 根据conf_thres进行过滤
    cfgs = [cfg for cfg in cfgs if cfg["score"] >= conf_thres]

    ## 根据focus_labels进行过滤
    cfgs = [cfg for cfg in cfgs if (focus_labels is None or cfg["label"] in focus_labels)]

    if len(cfgs) == 0:
        return []

    ## 根据diff_iou_thres进行过滤
    box_score = np.array([list(cfg["coor"])+[cfg["score"]] for cfg in cfgs], dtype=float)
    keep = nms(box_score, diff_iou_thres)
    cfgs = [cfgs[i] for i in keep]

    ## 根据same_iou_thres进行过滤
    label_dict = {}  # 根据labels对cfgs进行分类
    for cfg in cfgs:
        if cfg["label"] in label_dict:
            label_dict[cfg["label"]].append(cfg)
        else:
            label_dict[cfg["label"]] = [cfg]
    out_cfgs = []
    for label, cfgs in label_dict.items():
        box_score = np.array([list(cfg["coor"])+[cfg["score"]] for cfg in cfgs], dtype=float)
        keep = nms(box_score, same_iou_thres)
        out_cfgs = out_cfgs + [cfgs[i] for i in keep]
    cfgs = out_cfgs

    return cfgs
    

if __name__ == "__main__":
    candidates = selective_search("images/pointer/test_1.jpg")