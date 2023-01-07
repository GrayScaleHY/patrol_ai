import os
import time
import cv2
import json
from lib_image_ops import base642img, img2base64, img_chinese
from lib_help_base import color_list,GetInputData
from lib_sift_match import sift_match, convert_coor, sift_create
import numpy as np
import torch

from lib_inference_yolov5seg import load_yolov5seg_model, inference_yolov5seg, check_iou
from utils.segment.general import scale_image


## 加载模型
yolov5seg_daozha = load_yolov5seg_model("/data/PatrolAi/yolov5/daozha_seg.pt") # 加载油位的maskrcnn模型

def is_include(sub_box, par_box, srate=0.8):
    sb = sub_box;
    pb = par_box
    sb = [min(sb[0], sb[2]), min(sb[1], sb[3]), max(sb[0], sb[2]), max(sb[1], sb[3])]
    pb = [min(pb[0], pb[2]), min(pb[1], pb[3]), max(pb[0], pb[2]), max(pb[1], pb[3])]

    ## 至少一个点在par_box里面
    points = [[sb[0], sb[1]], [sb[2], sb[1]], [sb[0], sb[3]], [sb[2], sb[3]]]
    is_in = False
    for p in points:
        if p[0] >= pb[0] and p[0] <= pb[2] and p[1] >= pb[1] and p[1] <= pb[3]:
            is_in = True
    if not is_in:
        return False

    ## 判断交集占多少
    xmin = max(pb[0], sb[0]);
    ymin = max(pb[1], sb[1])
    xmax = min(pb[2], sb[2]);
    ymax = min(pb[3], sb[3])
    s_include = (xmax - xmin) * (ymax - ymin)
    s_box = (sb[2] - sb[0]) * (sb[3] - sb[1])
    if s_include / s_box >= srate:
        return True
    else:
        return False

'''def get_input_data(input_data):
    """
    提取input_data中的信息。
    return:
        img_tag: 目标图片数据
        img_ref: 模板图片数据
        roi: 感兴趣区域, 结构为[xmin, ymin, xmax, ymax]
    """

    img_tag = base642img(input_data["image"])

    ## 是否有模板图
    img_ref = None
    if "img_ref" in input_data["config"]:
        if isinstance(input_data["config"]["img_ref"], str):
            if len(input_data["config"]["img_ref"]) > 10:
                img_ref = base642img(input_data["config"]["img_ref"])

    #if img_ref==None:
    #    img_ref=img_tag.copy()

    ## 感兴趣区域
    roi = None  # 初始假设
    if "bboxes" in input_data["config"]:
        if isinstance(input_data["config"]["bboxes"], dict):
            if "roi" in input_data["config"]["bboxes"]:
                if isinstance(input_data["config"]["bboxes"]["roi"], list):
                    roi = input_data["config"]["bboxes"]["roi"]
                    dim = np.array(roi).ndim
                    if dim == 1:
                        if len(input_data["config"]["bboxes"]["roi"]) == 4:
                            W = img_ref.shape[1];
                            H = img_ref.shape[0]
                            roi = input_data["config"]["bboxes"]["roi"]
                            roi = [int(roi[0] * W), int(roi[1] * H), int(roi[2] * W), int(roi[3] * H)]
                    else:
                        for i in range(len(roi)):
                            if len(input_data["config"]["bboxes"]["roi"][i]) == 4:
                                W = img_ref.shape[1];
                                H = img_ref.shape[0]
                                roi[i]=[int(roi[i][0] * W), int(roi[i][1] * H), int(roi[i][2] * W), int(roi[i][3] * H)]

    if roi==[]:
        roi=None

    ## 设备状态与显示名字的映射关系。
    status_map = None
    if "status_map" in input_data["config"]:
        if isinstance(input_data["config"]["status_map"], dict):
            status_map = input_data["config"]["status_map"]

    ## 指定label_list。
    label_list = None
    if "label_list" in input_data["config"]:
        if isinstance(input_data["config"]["label_list"], list):
            label_list = input_data["config"]["label_list"]

    return img_tag, img_ref, roi, status_map, label_list
'''

def rankBbox(out_data_list,data_masks,roi,type='iou'):
    '''type:score bbox_size mask_size iou score&mask_size'''
    def cal_bbox_size(bbox):
        return abs((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))

    def cal_bbox_iou(rec1,rec2):
        if rec2==None:
            return abs((rec1[2] - rec1[0]) * (rec1[3] - rec1[1]))
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        sum_area = S_rec1 + S_rec2

        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])

        # judge if there is intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return (intersect*1.0 / (sum_area - intersect)) * 1.0

    if len(out_data_list)<=1:
        return out_data_list
    if type=='score':
        max_dict=out_data_list[0]
        for i in range(len(out_data_list)):
            if out_data_list[i]['score']>max_dict['score']:
                max_dict= out_data_list[i]
        return [max_dict]
    elif type=='bbox_size':
        max_dict = out_data_list[0]
        for i in range(len(out_data_list)):
            if cal_bbox_size(out_data_list[i]['bbox']) > cal_bbox_size(max_dict['bbox']):
                max_dict = out_data_list[i]
        return [max_dict]
    elif type=='score&mask_size':
        max_dict = out_data_list[0]
        max_mask_idx = 0
        for i in range(len(out_data_list)):
            if (data_masks[i].sum())*out_data_list[i]['score'] > (data_masks[max_mask_idx].sum())*max_dict['score']:
                max_dict = out_data_list[i]
                max_mask_idx = i
        return [max_dict]
    elif type=='iou':
        max_dict = out_data_list[0]
        for i in range(len(out_data_list)):
            if cal_bbox_iou(out_data_list[i]['bbox'],roi) > cal_bbox_iou(max_dict['bbox'],roi):
                max_dict = out_data_list[i]
        return [max_dict]
    else:
        max_dict = out_data_list[0]
        max_mask_idx=0
        for i in range(len(out_data_list)):
            if data_masks[i].sum() > data_masks[max_mask_idx].sum():
                max_dict = out_data_list[i]
                max_mask_idx=i
        return [max_dict]

def inspection_daozha_detection(input_data):
    """
    yolov5的目标检测推理。
    """
    ## 将输入请求信息可视化
    # TIME_START = time.strftime("%m%d%H%M%S") + "_"
    # if "checkpoint" in input_data and isinstance(input_data["checkpoint"], str) and len(input_data["checkpoint"]) > 0:
    #     TIME_START = TIME_START + input_data["checkpoint"] + "_"
    '''save_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_path = os.path.join(save_path, "result_patrol", input_data["type"])
    os.makedirs(save_path, exist_ok=True)
    f = open(os.path.join(save_path, TIME_START + "input_data.json"), "w")
    json.dump(input_data, f, ensure_ascii=False)  # 保存输入信息json文件
    f.close()'''

    ## 初始化输入输出信息。
    data = GetInputData(input_data)
    # img_tag, img_ref, roi, status_map, label_list = get_input_data(input_data)
    img_tag = data.img_tag
    img_ref = data.img_ref
    roi = data.roi
    checkpoint = data.checkpoint
    if roi==[]:
        roi=None

    out_data = {"code": 0, "data": [], "img_result": data.img_tag,
                "msg": "Success request object detect; "}  # 初始化out_data

    ## 将输入请求信息可视化
    img_tag_ = img_tag.copy()
    img_tag_ = img_chinese(img_tag_, checkpoint + data.type , (10, 10), color=(255, 0, 0), size=60)
    #cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag.jpg"), img_tag)  # 将输入图片可视化
    #if img_ref is not None:
        #cv2.imwrite(os.path.join(save_path, TIME_START + "img_ref.jpg"), img_ref)  # 将输入图片可视化
    if roi is not None:  # 如果配置了感兴趣区域，则画出感兴趣区域
        img_ref_ = img_ref.copy()
        dim = np.array(roi).ndim
        if dim == 1:
            if len(roi)==4:
                cv2.rectangle(img_ref_, (int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])), (255, 0, 255), thickness=1)
                cv2.putText(img_ref_, "roi", (int(roi[0]), int(roi[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255),
                    thickness=1)
                #cv2.imwrite(os.path.join(save_path, TIME_START + "img_ref_cfg.jpg"), img_ref_)
        else:
            for i in range(len(roi)):
                cv2.rectangle(img_ref_, (int(roi[i][0]), int(roi[i][1])), (int(roi[i][2]), int(roi[i][3])), (255, 0, 255),
                              thickness=1)
                cv2.putText(img_ref_, "roi"+str(i+1), (int(roi[i][0]), int(roi[i][1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 255),
                            thickness=1)
                #cv2.imwrite(os.path.join(save_path, TIME_START + "img_ref_cfg.jpg"), img_ref_)


    if "augm" in data.config:
        if isinstance(data.config["augm"], list):
            if len(data.config["augm"]) == 2:
                augm = data.config["augm"]
                augm = [float(augm[0]), float(augm[1])]
                img_tag = np.uint8(np.clip((augm[0] * img_tag + augm[1]), 0, 255))

    ## 生成目标检测信息
    labels=['budaowei','fen','he']
    cfgs = inference_yolov5seg(yolov5seg_daozha, img_tag, resize=640, pre_labels=labels, conf_thres=0.3)  # inference
    cfgs = check_iou(cfgs, iou_limit=0.5)  # 增加iou机制

    if len(cfgs) == 0:  # 没有检测到目标
        out_data["msg"] = out_data["msg"] + "; Not find object"
        out_data["code"] = 1
        out_data["data"]=[]
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=30)
        out_data["img_result"] = img2base64(img_tag_)
        #cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag_cfg.jpg"), img_tag_)
        return out_data

    ## labels 列表 和 color 列表
    colors = [(0,255,255),(0,255,0),(0,0,255)]#color_list(len(labels))
    color_dict = {}
    name_dict = {}
    for i, label in enumerate(labels):
        color_dict[label] = colors[i]
    name_dict = {'budaowei': '分合异常', 'fen': '分闸正常', 'he': '合闸正常'}

    ## 画出boxes
    #print(cfgs)
    for cfg in cfgs:
        c = cfg["coor"];
        label = cfg["label"]
        cv2.rectangle(img_tag_, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), color_dict[label], thickness=2)

        # Mask plotting
        masks = cfg["mask"]
        masks = masks.unsqueeze(0)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        masks = masks.permute(1, 2, 0).contiguous()
        masks = masks.cpu().numpy()
        masks = scale_image(masks.shape[:2], masks, img_tag_.shape)

        masks = np.asarray(masks, dtype=np.float32)
        alpha=0.3
        colors = np.asarray([color_dict[label]], dtype=np.float32)  # shape(n,3)
        print('shape', masks.shape, colors.shape)
        masks = (masks @ colors).clip(0, 255)  # (h,w,n) @ (n,3) = (h,w,3)
        s = masks.sum(2, keepdims=True).clip(0, 1)
        print(masks.shape,img_tag_.shape)
        img_tag_[:]= masks * alpha + img_tag_ * (1 - s * alpha)

        s = int((c[2] - c[0]) / 6)  # 根据框子大小决定字号和线条粗细。
        img_tag_ = img_chinese(img_tag_, name_dict[label], (c[0], c[1]), color=color_dict[label], size=s)

    ## 求出目标图像的感兴趣区域
    if roi is not None:
        feat_ref = sift_create(img_ref, rm_regs=[[0, 0, 1, 0.1], [0, 0.9, 1, 1]])
        feat_tag = sift_create(img_tag, rm_regs=[[0, 0, 1, 0.1], [0, 0.9, 1, 1]])
        M = sift_match(feat_ref, feat_tag, ratio=0.5, ops="Perspective")
        roi_tag=[]
        if M is None:
            out_data["msg"] = out_data["msg"] + "; Not enough matches are found"
            roi_tag = roi
        else:
            dim = np.array(roi).ndim
            if dim == 1:
                coors = [(roi[0], roi[1]), (roi[2], roi[1]), (roi[2], roi[3]), (roi[0], roi[3])]
                coors_ = []
                for coor in coors:
                    coors_.append(list(convert_coor(coor, M)))
                xs = [coor[0] for coor in coors_]
                ys = [coor[1] for coor in coors_]
                xmin = max(0, min(xs));
                ymin = max(0, min(ys))
                xmax = min(img_tag.shape[1], max(xs));
                ymax = min(img_tag.shape[0], max(ys))
                roi_tag=[xmin, ymin, xmax, ymax]
            else:
                for i in range(len(roi)):
                    coors = [(roi[i][0], roi[i][1]), (roi[i][2], roi[i][1]), (roi[i][2], roi[i][3]), (roi[i][0], roi[i][3])]
                    coors_ = []
                    for coor in coors:
                        coors_.append(list(convert_coor(coor, M)))
                    xs = [coor[0] for coor in coors_]
                    ys = [coor[1] for coor in coors_]
                    xmin = max(0, min(xs));
                    ymin = max(0, min(ys))
                    xmax = min(img_tag.shape[1], max(xs));
                    ymax = min(img_tag.shape[0], max(ys))
                    #roi_tag = [xmin, ymin, xmax, ymax]
                    roi_tag.append([xmin, ymin, xmax, ymax])

        ## 画出roi_tag
        c = roi_tag
        print('roi:',c)
        dim = np.array(c).ndim
        if dim == 1:
            cv2.rectangle(img_tag_, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (255, 0, 255), thickness=1)
            cv2.putText(img_tag_, "roi", (int(c[0]), int(c[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255),
                    thickness=1)
        else:
            for i in range(len(c)):
                #print(c[i])
                cv2.rectangle(img_tag_, (int(c[i][0]), int(c[i][1])), (int(c[i][2]), int(c[i][3])), (255, 0, 255), thickness=1)
                cv2.putText(img_tag_, "roi"+str(i), (int(c[i][0]), int(c[i][1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255),
                            thickness=1)

    ## 判断bbox是否在roi中 进行筛选
    bboxes = []
    out_data_data=[]
    data_masks=[]

    if roi is None:
        for cfg in cfgs:
            cfg_out = {"label": name_dict[cfg["label"]], "bbox": cfg["coor"], "score": float(cfg["score"])}
            out_data_data.append(cfg_out)
            data_masks.append(cfg["mask"])
            bboxes.append(cfg["coor"])
        out_data["data"] = rankBbox(out_data_data, data_masks, roi)
        cv2.putText(img_tag_, "-selected", (
                int(0.5 * out_data["data"][0]['bbox'][0] + 0.5 * out_data["data"][0]['bbox'][2]),
                int(out_data["data"][0]['bbox'][3]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), thickness=2)
    else:
        dim = np.array(roi_tag).ndim
        print('dim',dim)
        if dim == 1:
            for cfg in cfgs:
                if is_include(cfg["coor"], roi_tag, srate=0.5):
                    cfg_out = {"label": name_dict[cfg["label"]], "bbox": cfg["coor"], "score": float(cfg["score"])}
                    out_data_data.append(cfg_out)
                    data_masks.append(cfg["mask"])
                    bboxes.append(cfg["coor"])
            out_data["data"] = rankBbox(out_data_data, data_masks, roi)
            cv2.putText(img_tag_, "-selected", (
                int(0.5 * out_data["data"][0]['bbox'][0] + 0.5 * out_data["data"][0]['bbox'][2]),
                int(out_data["data"][0]['bbox'][3]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), thickness=2)
        else:
            for i in range(len(roi_tag)):
                bboxes = []
                out_data_data = []
                data_masks = []
                for cfg in cfgs:
                    if is_include(cfg["coor"], roi_tag[i], srate=0.5):
                        cfg_out = {"label": name_dict[cfg["label"]], "bbox": cfg["coor"], "score": float(cfg["score"])}
                        out_data_data.append(cfg_out)
                        data_masks.append(cfg["mask"])
                        bboxes.append(cfg["coor"])
                print(out_data_data)
                tmp_data=rankBbox(out_data_data, data_masks, roi[i])
                print(tmp_data)
                if tmp_data!=[]:
                    out_data["data"].append(tmp_data[0])
                    cv2.putText(img_tag_, "-selected"+str(i), (
                        int(0.5 * tmp_data[0]['bbox'][0] + 0.5 * tmp_data[0]['bbox'][2]),
                        int(tmp_data[0]['bbox'][3]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), thickness=2)

    ## 可视化计算结果
    '''f = open(os.path.join(save_path, TIME_START + "out_data.json"), "w")
    json.dump(out_data, f, ensure_ascii=False, indent=2)  # 保存输入信息json文件
    f.close()'''
    #cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag_cfg.jpg"), img_tag_)
    #cv2.imwrite(os.path.join( TIME_START + "img_tag_cfg.jpg"), img_tag_)

    ## 输出可视化结果的图片。
    out_data["img_result"] = img2base64(img_tag_)

    return out_data


if __name__ == '__main__':
    json_file = "1223110334_input_data.json"
    f = open(json_file, "r", encoding='utf-8')
    input_data = json.load(f)
    f.close()
    out_data = inspection_daozha_detection(input_data)
    print("inspection_daozha_detection result:")
    print("-----------------------------------------------")
    for s in out_data:
        if s != "img_result":
            print(s, ":", out_data[s])
    print("----------------------------------------------")
