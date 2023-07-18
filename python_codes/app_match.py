from lib_help_base import GetInputData
from lib_img_registration import registration, convert_coor
import numpy as np
import cv2
import os
import wget

def lib_match(img_ref, img_tag, pointers, bboxes):
    H, W = img_tag.shape[:2]
    M = registration(img_ref, img_tag)

    out_pointers = {}
    for scale in pointers:
        coor = convert_coor(pointers[scale], M)
        out_pointers[scale] = (coor[0] / W, coor[1] / H)
    
    out_bboxes = {}
    for label in bboxes:
        box = bboxes[label]
        coors = [(box[0],box[1]), (box[2],box[1]), (box[2],box[3]), (box[0],box[3])]
        coors_ = [list(convert_coor(coor, M)) for coor in coors]
        c_ = np.array(coors_, dtype=int)
        H, W = img_tag.shape[:2]
        r = [min(c_[:,0]), min(c_[:, 1]), max(c_[:,0]), max(c_[:,1])]
        out_bboxes[label] = [max(0, r[0])/W, max(0, r[1])/H, min(W, r[2])/W, min(H, r[3])/H]
    
    return out_pointers, out_bboxes

def get_img_paths(input_data):
    img_paths = []
    for tag_ in input_data["images_path"]:
        if os.path.exists(tag_):
            img_paths.append(tag_)
        elif tag_.startswith("http"):
            tag_tmp = "/export" + tag_.split(":9000")[1]
            tag_dir = os.path.dirname(tag_tmp)
            os.makedirs(tag_dir, exist_ok=True)
            wget.download(tag_, tag_tmp)
            img_paths.append(tag_tmp)
        else:
            continue
    return img_paths


def patrol_match(input_data):
    """
    目标追踪接口
    https://git.utapp.cn/xunshi-ai/json-http-interface/-/wikis/模板配置-目标追踪-(1对多)
    """
    ## 提取输入请求信息
    DATA = GetInputData(input_data)
    img_tag = DATA.img_tag
    img_ref = DATA.img_ref
    pointers = DATA.pointers #{"0": [1500, 2100],"1":[1680, 780]}
    bboxes = DATA.bboxes
    an_type = DATA.type

    if an_type == "track":
        out_pointers, out_bboxes = lib_match(img_ref, img_tag, pointers, bboxes)
        out_data = {"code":0, "data": {"pointers": out_pointers, "bboxes": out_bboxes}, "msg": "Success!"}
        return out_data

    out_data = {"code":0, "data": [], "msg": "Success!"}
    for tag_ in get_img_paths(input_data):
        img_tag = cv2.imread(tag_)
        out_pointers, out_bboxes = lib_match(img_ref, img_tag, pointers, bboxes)
        out_data["data"].append({"pointers": out_pointers, "bboxes": out_bboxes})

    return out_data


if __name__ == '__main__':
    import json
    json_file = "/data/PatrolAi/result_patrol/0718085932_input_data.json"
    print(json_file)
    f = open(json_file,"r", encoding='utf-8')
    input_data = json.load(f)
    f.close()
    out_data = patrol_match(input_data)

    print("------------------------------")
    for s in out_data:
        if s != "img_result":
            print(s, ":", out_data[s])
    print("------------------------------")
    
