from lib_help_base import GetInputData
from lib_img_registration import registration, convert_coor
import numpy as np
import cv2
import os
import requests
# import wget

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
            print("request download--------------------------------------")
            print(tag_)
            print(tag_tmp)
            os.makedirs(tag_dir, exist_ok=True)
            r = requests.get(tag_)
            f = open(tag_tmp, "wb")
            f.write(r.content)
            f.close()
            # wget.download(tag_, tag_tmp)
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
        try:
            out_pointers, out_bboxes = lib_match(img_ref, img_tag, pointers, bboxes)
        except:
            print("WARNING: wrong file:", tag_)
            out_pointers = input_data["config"]["pointers"]
            out_bboxes =    input_data["config"]["bboxes"]   
            
        out_data["data"].append({"pointers": out_pointers, "bboxes": out_bboxes})

    return out_data


if __name__ == '__main__':
    import json
    # json_file = "/data/PatrolAi/result_patrol/0718085932_input_data.json"
    # print(json_file)
    # f = open(json_file,"r", encoding='utf-8')
    # input_data = json.load(f)
    # f.close()
    # pointers_abs = {"center": [0.1, 0.2], "-0.1": [0.3, 0.4], "0.9": [0.5, 0.6]}
    xml_file = "/data/PatrolAi/patrol_ai/python_codes/images/00031.xml"
    print(xml_file)
    pointers = {}
    for line in open(xml_file, "r", encoding='utf-8'):
        if "<width>" in line:
            W = int(line.split("width>")[1][:-2])
        if "<height>" in line:
            H = int(line.split("height>")[1][:-2])
        if "<name>" in line:
            label = line.split("name>")[1][:-2]
        if "<xmin>" in line:
            xmin = int(line.split("xmin>")[1][:-2])
        if "<ymin>" in line:
            ymin = int(line.split("ymin>")[1][:-2])
            point = [xmin / W, ymin / H]
            pointers[label] = point
            # import pdb
            # pdb.set_trace()
        
    input_data = {
        "images_path": [
            '/data/PatrolAi/patrol_ai/python_codes/images/0003 (1).jpg',
            '/data/PatrolAi/patrol_ai/python_codes/images/0003 (2).jpg',
            '/data/PatrolAi/patrol_ai/python_codes/images/0003 (3).jpg',
            '/data/PatrolAi/patrol_ai/python_codes/images/0003 (4).jpg',
            '/data/PatrolAi/patrol_ai/python_codes/images/0003 (5).jpg',
            '/data/PatrolAi/patrol_ai/python_codes/images/0003 (6).jpg'
        ],
        "config": {
            "img_ref_path": "/data/PatrolAi/patrol_ai/python_codes/images/0003 (1).jpg",
            "pointers": pointers,
            "bboxes": []
        },
        "type": "track_batch"
    }
    out_data = patrol_match(input_data)

    print(out_data)

    for i, img_file in enumerate(input_data["images_path"]):
        img = cv2.imread(img_file)
        H, W = img.shape[:2]
        pointers = out_data["data"][i]["pointers"]
        for label in pointers:
            coor = (int(pointers[label][0] * W), int(pointers[label][1] * H))
            cv2.circle(img, (int(coor[0]), int(coor[1])), 1, (255, 0, 255), 8)
            cv2.putText(img, label, (int(coor[0]), int(
                coor[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)
        
        cv2.imwrite(os.path.join("images_sel", os.path.basename(img_file)), img)


    

    
