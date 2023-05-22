from lib_help_base import GetInputData
from lib_img_registration import registration, convert_coor
import numpy as np
import cv2
from lib_image_ops import img2base64

def patrol_match(input_data):
    ## 提取输入请求信息
    DATA = GetInputData(input_data)
    img_tag = DATA.img_tag
    img_ref = DATA.img_ref
    pointers = DATA.pointers #{"0": [1500, 2100],"1":[1680, 780]}
    bboxes = DATA.bboxes
    M = registration(img_ref, img_tag)

    out_pointers = {}
    for scale in pointers:
        coor = pointers[scale]
        out_pointers[scale] = convert_coor(coor, M)
    
    out_bboxes = {}
    for label in bboxes:
        box = bboxes[label]
        coors = [(box[0],box[1]), (box[2],box[1]), (box[2],box[3]), (box[0],box[3])]
        coors_ = [list(convert_coor(coor, M)) for coor in coors]
        c_ = np.array(coors_, dtype=int)
        H, W = img_tag.shape[:2]
        r = [min(c_[:,0]), min(c_[:, 1]), max(c_[:,0]), max(c_[:,1])]
        box_ = [max(0, r[0]), max(0, r[1]), min(W, r[2]), min(H, r[3])]
        out_bboxes[label] = box_
    
    img_tag_ = img_tag.copy()
    for scale in out_pointers:  # 将坐标点标注在图片上
        coor = out_pointers[scale]
        cv2.circle(img_tag_, (int(coor[0]), int(coor[1])), 1, (255, 0, 255), 8)
        cv2.putText(img_tag_, str(scale), (int(coor[0]), int(coor[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)
    
    for label in out_bboxes:
        o_ = out_bboxes[label]
        cv2.rectangle(img_ref, (int(o_[0]), int(o_[1])),(int(o_[2]), int(o_[3])), (255, 0, 255), thickness=1)
        cv2.putText(img_ref, label, (int(o_[0]), int(o_[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)

    out_data = {"code":0, "data": {"pointers": out_pointers, "bboxes": out_bboxes}, "img_result":img2base64(img_tag_), "msg": "Success!"}
    return out_data
    
