import sys
import random
import sympy
import math
from config_object_name import COLOR_HSV_MAP
import cv2
import numpy as np
from lib_image_ops import base642img, img2base64, img_chinese
import json
import os
import time
import shutil
import wget
import requests
import glob

class GetInputData:
    """
    获取巡视输入信息。
    """

    def __init__(self, data):
        # self.input_data = input_data
        self.checkpoint = self.get_checkpoint(data)  # 点位名称
        self.img_tag = self.get_img_tag(data)  # 测试图
        self.type = self.get_type(data)  # 分析类型
        self.config = self.get_config(data)  # 模板信息
        self.video_path = self.get_video_path(data)
        self.img_ref = self.get_img_ref(self.config) # 模板图
        self.roi = self.get_roi(self.config, self.img_ref)  # roi框
        self.regbox=self.get_regbox(self.config)
        self.pointers = self.get_pointers(self.config, self.img_ref)  # 刻度点坐标信息
        self.bboxes = self.get_bboxes(self.config, self.img_ref) # 目标框坐标信息
        self.osd = self.get_osd(self.config, self.img_ref) # osd框
        self.dp = self.get_dp(self.config)
        self.dp_dict = self.get_dpdict(self.config)
        self.number, self.length, self.width, self.color, self.val_size, self.meter_type = self.get_pointer_cfg(self.config) # 多指针的性质
        self.status_map = self.get_status_map(self.config)
        self.label_list = self.get_label_list(self.config)
        self.sense = self.get_sense(self.config)
        self.is_region = self.get_region(self.config)
        self.fineness = self.get_fineness(self.config)

        
    def get_checkpoint(self, data):
        """
        获取checkpoint(巡检点位名称)。
        """
        if "checkpoint" in data and isinstance(data["checkpoint"], str):
            checkpoint = data["checkpoint"]
        else:
            checkpoint = ""
        return checkpoint
    
    def get_img(self, img_str):
        """
        将字符串形式的图片转成numpy图片，字符串形式图片可以是base64格式、http链接、图片绝对路径
        """
        if img_str.startswith("http"): # 如果是网络图，则尝试下载
            img_str_file = "/export/" + "/".join(img_str.split("/")[-3:])
            if not os.path.exists(img_str_file): # 如果不存在，则下载
                img_str_dir = os.path.dirname(img_str_file)
                os.makedirs(img_str_dir, exist_ok=True)
                print("request download--------------------------------------")
                print(img_str_file)
                r = requests.get(img_str)
                f = open(img_str_file, "wb")
                f.write(r.content)
                f.close()
                # wget.download(img_ref_path, img_ref_path_)
            img = cv2.imread(img_str_file)

        elif os.path.exists(img_str): # 如果是绝对路径，则直接读取
            img = cv2.imread(img_str)

        else: # 若是base64,转换为img
            try:
                img = base642img(img_str)
            except:
                print("data[image] can not convert to img!")
                img = None

        return img
    
    def get_img_tag(self, data):
        """
        获取测试图
        return:
            img_tag: numpy的图像数据。
        """
        if "image" not in data or not isinstance(data["image"], str):
            return None
        img_t = data["image"]
        
        return self.get_img(img_t)
    
    def get_type(self, data):
        """
        获取分析类型。
        """
        if "type" in data and isinstance(data["type"], str) and data['type'] != "":
            type_ = data["type"]
        else:
            type_ = None
        return type_
    
    def get_config(self, data):
        """
        获取模板信息。
        """
        if "config" in data and isinstance(data["config"], dict):
            config = data["config"]
        else:
            config = {}
        return config

    def get_img_ref(self, config):
        """
        获取模板图。
        """
        if "img_ref" in config and isinstance(config["img_ref"], str):
            img_r = config["img_ref"]
        elif "img_ref_path" in config and isinstance(config["img_ref_path"], str):
            img_r = config["img_ref_path"]
        else:
            return None
        
        return self.get_img(img_r)
    
    def get_roi(self, config, img_ref):
        """
        获取roi感兴趣区域
        return:
            roi: {"yb1": [xmin, ymin, xmax, ymax], ..}
        """
        if img_ref is None:
            return {}

        if "bboxes" not in config or not isinstance(config["bboxes"], dict) or "roi" not in config["bboxes"]:
            return {}
        
        raw_roi = config["bboxes"]["roi"]

        if not isinstance(raw_roi, list) and not isinstance(raw_roi, dict):
            return {}

        ## 老版本的roi是list，将list改为dict
        if isinstance(raw_roi, list):
            if len(raw_roi) == 0:
                raw_roi = {}
            else:
                _raw_roi = raw_roi
                dim = np.array(_raw_roi).ndim
                if dim == 1:
                    _raw_roi = [_raw_roi]
                raw_roi = {}
                for i in range(len(_raw_roi)):
                    raw_roi["old_roi" + str(i)] = _raw_roi[i]
        
        ## 将raw_roi中的坐标改为绝对坐标
        H, W = img_ref.shape[:2]
        roi = {}
        for name in raw_roi:
            _roi = raw_roi[name]
            roi[name] = [int(float(_roi[0])*W), int(float(_roi[1])*H), int(float(_roi[2])*W), int(float(_roi[3])*H)]
        
        return roi

    def get_regbox(self,config):
        """
            获取纠偏区域框
            return:
                [xmin, ymin, xmax, ymax]或[]
        """
        try:
            if config["bboxes"]["reg"]==[]:
                return []
        except :
            return []
        return config["bboxes"]["reg"]

    
    def get_osd(self, config, img_ref):
        """
        获取osd区域
        return:
            osd: 格式为二维列表[[xmin, ymin, xmax, ymax], ..], 或者空列表[]
        """
        if img_ref is None:
            return []

        if "bboxes" in config and isinstance(config["bboxes"], dict):
            bboxes = config["bboxes"]
        else:
            bboxes = {}

        if "osd" in bboxes and isinstance(bboxes["osd"], list) and len(bboxes["osd"]) > 0:
            raw_osd = bboxes["osd"]
        else:
            return []
        
        dim = np.array(raw_osd).ndim
        if dim == 1:
            raw_osd = [raw_osd]
        
        osd = []
        H, W = img_ref.shape[:2]
        for _osd in raw_osd:
            osd.append([int(_osd[0]*W), int(_osd[1]*H), int(_osd[2]*W), int(_osd[3]*H)])
        
        return osd
    
    def get_pointers(self, config, img_ref):
        """
        获取仪表各刻度的位置。
        """
        if img_ref is None:
            return {}

        if "pointers" in config and isinstance(config["pointers"], dict):
            raw_pointers = config["pointers"]
        else:
            raw_pointers = {}

        H, W = img_ref.shape[:2]
        pointers = {}
        for scale in raw_pointers:
            point = raw_pointers[scale]
            pointers[scale] = [int(float(point[0]) * W), int(float(point[1]) * H)]
        
        return pointers
    
    def get_bboxes(self, config, img_ref):
        """
        获取目标框的位置坐标。
        """
        if img_ref is None:
            return {}

        if "bboxes" in config and isinstance(config["bboxes"], dict):
            bboxes_ = config["bboxes"]
            raw_bboxes = {}
            for name in bboxes_:
                if isinstance(bboxes_[name], list):
                    raw_bboxes[name] = bboxes_[name]
        else:
            raw_bboxes = {}

        H, W = img_ref.shape[:2]
        bboxes = {}
        for label in raw_bboxes:
            bbox = raw_bboxes[label]
            dim = np.array(bbox).ndim
            if dim == 2:
                bbox = bbox[0]
            bboxes[label] = [int(bbox[0] * W), int(bbox[1] * H), int(bbox[2] * W), int(bbox[3] * H)]
        
        return bboxes
    
    def get_dp(self, config):
        """
        获取数值小数点位数
        """
        if "dp" in config and isinstance(config["dp"], int) and config["dp"] != -1:
            dp = int(config["dp"])
        else:
            dp = 3
        return dp

    def get_dpdict(self, config):
        """
        获取数值小数点位数列表
        """
        if "dp_dict" in config and isinstance(config["dp_dict"], dict):
            dp_dict = config["dp_dict"]
        else:
            dp_dict = {}
        return dp_dict
    
    def get_pointer_cfg(self, config):
        """
        获取指针个数, 长短， 宽窄， 颜色。
        """
        if "number" in config and isinstance(config["number"], int) and config["number"] != -1:
            number = int(config["number"])
        else:
            number = 1

        if "length" in config and isinstance(config["length"], int) and config["length"] != -1:
            length = int(config["length"])
        else:
            length = None
        
        if "width" in config and isinstance(config["width"], int) and config["width"] != -1:
            width = int(config["width"])
        else:
            width = None

        if "color" in config and isinstance(config["color"], int) and config["color"] != -1:
            color = int(config["color"])
        else:
            color = None
        
        if "val_size" in config and isinstance(config["val_size"], int) and config["val_size"] != -1:
            val_size = int(config["val_size"])
        else:
            val_size = None
        
        if "meter_type" in config and isinstance(config["meter_type"], str) and len(config["meter_type"]) > 1:
            if config["meter_type"] == "blq_zzscsb":
                meter_type = "blq_zzscsb"
            elif config["meter_type"] == "nszb":
                meter_type = "nszb"
            elif config["meter_type"] == "nszb_zzscsb":
                meter_type = "nszb_zzscsb"
            else: 
                meter_type = "normal"
        else:
            meter_type = "normal"
        
        return number, length, width, color, val_size, meter_type

    def get_status_map(self, config):
        if "status_map" in config and isinstance(config["status_map"], dict):
            status_map = config["status_map"]
        else:
            status_map = {}
        return status_map

    def get_label_list(self, config):
        if "label_list" in config and isinstance(config["label_list"], list):
            label_list = config["label_list"]
        else:
            label_list = []
        return label_list

    def get_sense(self, config):
        if "sense" in config and isinstance(config["sense"], (int, float)) and config["sense"] > 0:
            sense = config["sense"]
        else:
            sense = None
        return sense

    def get_fineness(self, config):
        if "fineness" in config and isinstance(config["fineness"], (int, float)) and config["fineness"] >= 0:
            fineness = config["fineness"]
        else:
            fineness = 5
        return fineness
    
    def get_region(self, config):
        if "is_region" in config and isinstance(config["is_region"], (int, float)):
            is_region = config["is_region"]
        else:
            is_region = 0
        return is_region
    
    def get_video_path(self, data):
        """
        获取测试视频路径
        """
        if "video_path" in data and isinstance(data["video_path"], str):

            if  os.path.exists(data["video_path"]): #  and 
                if not data["video_path"].endswith(".mp4"):
                    print("Warning: video_path is not .mp4!")
                video_path = data["video_path"]
            else:
                
                assert data["video_path"].startswith("http"), data["video_path"] + "is not http url !"

                video_path = "/export" + data["video_path"].split(":9000")[1]
                if os.path.exists(video_path):
                    return video_path
                else:
                    video_dir = os.path.dirname(video_path)
                    os.makedirs(video_dir, exist_ok=True)
                    print("request download--------------------------------------")
                    print(data["video_path"])
                    print(video_path)
                    # r = requests.get(data["video_path"])
                    # f = open(video_path, "wb")
                    # f.write(r.content)
                    # f.close()
                    wget.download(data["video_path"], video_path)
                
        else:
            video_path = ""


        return video_path

class Logger(object):
    """
    将控制端log保存下来的方法。
    demo:
        sys.stdout = Logger("log.txt")
    """
    def __init__(self,fileN ="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN,"a")
 
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

def color_list(c_size):
    """
    生成一个颜色列表。
    """
    color_map = {
        # 0: (0, 0, 0), # 黑色
        # 1: (255, 255, 255), # 白色
        0: (0, 0, 255), # 红色
        1: (0, 255, 0), # 绿色
        2: (255, 0, 0), # 蓝色
        3: (0, 255, 255), # 黄色
        4: (255, 0, 255), # 粉色
        5: (255, 255, 0), # 淡蓝
    }
    colors = []
    if c_size <= len(color_map):
        for i in range(c_size):
            colors.append(color_map[i])
    else:
        colors = [color_map[c] for c in color_map]
        for i in range(c_size - len(color_map)):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            colors.append(color)
    return colors

def oil_high(s_oil, s_round):
    """
    根据油面积和圆面积求油位的高度
    args:
        s_oil: 油面积
        s_round: 圆面积
    return:
        h: 油位高度
    """
    R = math.sqrt(s_round / math.pi) #根据圆面积求圆半径
    S = s_oil

    if S > s_round / 2: # 油位在圆心之上
        d=sympy.Symbol('H') # 圆心到油位的距离

        ## 大扇形面积加上等腰三角形面积等于油面积
        fx = (sympy.pi - sympy.acos(d/R))*R**2  + R * sympy.cos(sympy.asin(d/R)) * d - S
        d = sympy.nsolve(fx,d,0) # 赋值解方程
        h = d + R

    elif S < s_round / 2: # 油位在圆心之下
        d=sympy.Symbol('H')

        ## 小扇形面积减去等腰三角形面积等于油面积
        fx = sympy.acos(d/R)*R**2  - R * sympy.cos(sympy.asin(d/R)) * d - S
        d = sympy.nsolve(fx,d,0) # 赋值解方程
        h =  R - d

    else:
        h = R

    return h

def color_area(img, color_list=["black","white","red","red2","orange","yellow","green","cyan","blue","purple"]):
    """
    根据hsv颜色空间判断图片中各颜色的面积。
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_dict = {}
    for color in color_list:
        hsv_lower = np.array(COLOR_HSV_MAP[color][0])
        hsv_upper = np.array(COLOR_HSV_MAP[color][1])
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

        # color_sum = np.sum(mask)

        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary,None,iterations=2)
        cnts, hiera = cv2.findContours(binary.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        color_sum = 0
        for c in cnts:
            color_sum += cv2.contourArea(c)

        color_dict[color] = color_sum
    
    return color_dict

def get_save_head(input_data):
    """
    根据input_data和当前时刻获取保存文件夹和文件名头
    args:
        input_data: 巡视算法输入数据
    return:
        save_dir: 保存的文件夹路径
        name_head: 文件名的开头
    """
    save_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_dir = os.path.join(save_dir, "result_patrol", input_data["type"])
    os.makedirs(save_dir, exist_ok=True)
    name_head = time.strftime("%m%d%H%M%S") + "_"
    if "checkpoint" in input_data and isinstance(input_data["checkpoint"], str):
        name_head = name_head + input_data["checkpoint"] + "_"
        if "/" in name_head:
            name_head = name_head.replace("/", "")

    return save_dir, name_head


def save_input_data(input_data, save_dir, name_head, draw_img=False):
    """
    保存巡视算法输入数据
    args:
        input_data: 巡视传过来的输入数据
        save_dir: 保存的文件夹路径
        name_head: 文件名的开头
        draw_img： 是否画图，if False: 只保存input_data; if True: 将信息画在图上
    """

    f = open(os.path.join(save_dir, name_head + "input_data.json"), "w", encoding='utf-8')
    json.dump(input_data, f, ensure_ascii=False)  # 保存输入信息json文件
    f.close()

    if not draw_img:
        return 0

    DATA = GetInputData(input_data)
    img_tag = DATA.img_tag; img_ref = DATA.img_ref
    pointers_ref = DATA.pointers
    roi = DATA.roi; osd = DATA.osd
    video_path = DATA.video_path
    bboxes_ref = DATA.bboxes
    
    if img_tag is not None:
        cv2.imwrite(os.path.join(save_dir, name_head + "tag.jpg"), img_tag)
    else:
        return 0
        
    if img_ref is not None:
        cv2.imwrite(os.path.join(save_dir, name_head + "ref.jpg"), img_ref)
    else:
        return 0
    
    if os.path.exists(video_path):
        shutil.copy(video_path, os.path.join(save_dir, name_head + "tag.mp4"))

    for scale in pointers_ref:  # 将坐标点标注在图片上
        coor = pointers_ref[scale]
        cv2.circle(img_ref, (int(coor[0]), int(coor[1])), 1, (255, 0, 255), 8)
        cv2.putText(img_ref, str(scale), (int(coor[0]), int(coor[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)
    for name, o_ in roi.items():   ## 如果配置了感兴趣区域，则画出感兴趣区域
        cv2.rectangle(img_ref, (int(o_[0]), int(o_[1])),(int(o_[2]), int(o_[3])), (255, 0, 255), thickness=1)
        cv2.putText(img_ref, name, (int(o_[0]), int(o_[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)
    for o_ in osd:  ## 如果配置了感兴趣区域，则画出osd区域
        cv2.rectangle(img_ref, (int(o_[0]), int(o_[1])),(int(o_[2]), int(o_[3])), (255, 0, 255), thickness=1)
        cv2.putText(img_ref, "osd", (int(o_[0]), int(o_[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)
    for label in bboxes_ref:
        if label == "roi":
            continue
        o_ = bboxes_ref[label]
        cv2.rectangle(img_ref, (int(o_[0]), int(o_[1])),(int(o_[2]), int(o_[3])), (255, 0, 255), thickness=1)
        cv2.putText(img_ref, label, (int(o_[0]), int(o_[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)
    cv2.imwrite(os.path.join(save_dir, name_head + "ref_cfg.jpg"), img_ref)

def save_output_data(output_data, save_dir, name_head):
    """
    保存巡视算法输入数据
    args:
        output_data: 算法分析的结果
        save_dir: 保存的文件夹路径
        name_head: 文件名的开头
    """
    f = open(os.path.join(save_dir, name_head + "output_data.json"), "w", encoding='utf-8')
    json.dump(output_data, f, ensure_ascii=False)  # 保存输入信息json文件
    f.close()
    if os.path.exists(output_data["img_result"]):
        shutil.copy(output_data["img_result"], os.path.join(save_dir, name_head + "tag_cfg.jpg"))
    else:
        img_tag_cfg = base642img(output_data["img_result"])
        cv2.imwrite(os.path.join(save_dir, name_head + "tag_cfg.jpg"), img_tag_cfg)

def is_include(sub_box, par_box, srate=0.8):
    
    sb = sub_box; pb = par_box
    sb = [min(sb[0],sb[2]), min(sb[1],sb[3]), max(sb[0],sb[2]), max(sb[1],sb[3])]
    pb = [min(pb[0],pb[2]), min(pb[1],pb[3]), max(pb[0],pb[2]), max(pb[1],pb[3])]

    ## 至少两个框需要有交集
    minx = max(sb[0], pb[0]); maxx = min(sb[2], pb[2])
    miny = max(sb[1], pb[1]); maxy = min(sb[3], pb[3])
    if minx > maxx or miny > maxy:
        return False
 
    ## 判断交集占多少
    xmin = max(pb[0], sb[0]); ymin = max(pb[1], sb[1])
    xmax = min(pb[2], sb[2]); ymax = min(pb[3], sb[3])
    s_include = (xmax-xmin) * (ymax-ymin)
    s_box = (sb[2]-sb[0]) * (sb[3]-sb[1])
    if s_include / s_box >= srate:
        return True
    else:
        return False

def creat_img_result(input_data, img_tag_):
    """
    巡视算法输出的img_result兼容图片路径和base64.
    """
    if os.path.exists(input_data["image"]): 
        out_file = input_data["image"][:-4] + "_result.jpg"
        cv2.imwrite(out_file, img_tag_)
        img_result = out_file
    else:
        img_result = img2base64(img_tag_)
    return img_result

def draw_region_result(out_data, input_data, roi_tag):
    """
    每个区域返回一张结果展示图
    """
    DATA = GetInputData(input_data)
    is_region = DATA.is_region
    checkpoint = DATA.checkpoint
    img_tag = DATA.img_tag
    an_type = DATA.type
    
    if not is_region:
        return out_data
    
    for name in out_data["data"]:
        img_tag_ = img_tag.copy()
        ## 画上点位名称
        img_tag_ = img_tag.copy()
        img_tag_ = img_chinese(img_tag_, an_type + "_" + name , (10, 100), color=(255, 0, 0), size=30)
        c = roi_tag[name]
        cv2.rectangle(img_tag_, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), (255,0,255), thickness=1)
        # cv2.putText(img_tag_, name, (int(c[0]), int(c[1])+10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)
        s = int((c[2] - c[0]) / 10) # 根据框子大小决定字号和线条粗细。
        img_tag_ = img_chinese(img_tag_, name, (c[0], c[1]), color=(255,0,255), size=s)
        for cfg in out_data["data"][name]:
            if "bbox" in cfg:
                b = cfg["bbox"]
                # 画出识别框
                cv2.rectangle(img_tag_, (int(b[0]), int(b[1])),(int(b[2]), int(b[3])), (0,0,255), thickness=2)
            else:
                b = c

            s = int((b[2] - b[0]) / 6) # 根据框子大小决定字号和线条粗细。
            if "label" in cfg:
                label = cfg["label"]
            elif "values" in cfg:
                label = cfg["values"]
            else:
                label = ""
            img_tag_ = img_chinese(img_tag_, label, (b[0], b[1]), color=(0,0,255), size=s)
        
        for i in range(len(out_data["data"][name])):
            # if len(out_data["data"][name][i]) == 0:
            #     continue
            if os.path.exists(input_data["image"]): 
                out_file = input_data["image"][:-4] + "_" + name + "_result.jpg"
                cv2.imwrite(out_file, img_tag_)
                img_result = out_file
            else:
                img_result = img2base64(img_tag_)
            
            out_data["data"][name][i]["img"] = img_result
    
    return out_data

def traverse_and_modify(obj):
    """
    如果字典中的val是字符串且长度大于200，则改为字符串改为base64_image.
    """
    if isinstance(obj, dict):
        new_obj = {}
        for key, value in obj.items():
            new_value = traverse_and_modify(value)
            new_obj[key] = new_value
        return new_obj
    elif isinstance(obj, list):
        return [traverse_and_modify(item) for item in obj]
    elif isinstance(obj, str) and len(obj) > 200: # 如果字符长度大于20
        return "base64_image"  # 字符变为base64_image
    else:
        return obj
    

def rm_patrolai(out_json, save_dict, out_dir):
    """
    将分析结果保存下来的图片删除或转移
    """
    out_name = os.path.basename(out_json)
    file_head = out_json[:-16]
    checkpoint = out_name[11:-16]
    type_ = os.path.basename(os.path.dirname(out_json))
    rm_list = []
    rm_list.append(out_json)
    rm_list.append(file_head + "input_data.json")
    rm_list.append(file_head + "tag.jpg")
    rm_list.append(file_head + "tag_cfg.jpg")
    rm_list.append(file_head + "ref.jpg")
    rm_list.append(file_head + "ref_cfg.jpg")
    rm_list.append(file_head + "tag.mp4")

    f = open(out_json, "r", encoding='utf-8')
    out_data = json.load(f)
    f.close()
    if out_data["code"] == 1:
        is_cp = True
    else:
        is_cp = False
        if "data" in out_data and isinstance(out_data["data"], list):
            labels = []
            for cfg in out_data["data"]:
                if "label" in cfg:
                    labels.append(cfg["label"])
            labels.sort()

            if len(labels) > 0: #and labels not in save_dict[type_][checkpoint]:
                if type_ in save_dict and checkpoint in save_dict[type_]:
                    if labels not in save_dict[type_][checkpoint]["labels"]:
                        save_dict[type_][checkpoint]["labels"].append(labels)
                        print(type_ + "/" + checkpoint + ":" + str(save_dict[type_][checkpoint]))
                        is_cp = True
                else:
                    if type_ not in save_dict:
                        save_dict[type_] = {}
                    if checkpoint not in save_dict[type_]:
                        save_dict[type_][checkpoint] = {"labels": [labels]}
                        is_cp = True

    file_name = os.path.basename(out_json)
    day_head = float(file_name[:4])

    for in_file in rm_list:
        if not os.path.exists(in_file):
            continue
        out_file = os.path.join(out_dir, type_, os.path.basename(in_file))
        if is_cp and not os.path.exists(out_file) and out_file.endswith(".json"):
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            shutil.copy(in_file, out_file)
        
        day_now = float(time.strftime("%m%d"))
        if day_head > day_now or day_head < day_now - 1:
            os.remove(in_file)
    
    return save_dict


def rm_result_patrolai():
    """
    将分析异常的图片转移到/data/PatrolAi/result_saved
    """
    
    in_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    in_dir = os.path.join(in_dir, "result_patrol")

    while True:

        month = time.strftime("%Y%m")
        out_dir = os.path.join(os.path.dirname(in_dir), "result_saved", month)
        os.makedirs(out_dir, exist_ok=True)

        ## 初始化save_dict
        save_file = os.path.join(out_dir, "save_dict.json")
        if os.path.exists(save_file):
            f = open(save_file, "r", encoding='utf-8')
            save_dict = json.load(f)
            f.close()
        else:
            json_list = glob.glob(os.path.join(os.path.dirname(in_dir), "result_saved/*/save_dict.json"))
            json_list.sort()
            if len(json_list) < 1:
                save_dict = {}
            else:
                json_file = json_list[-1]
                f = open(json_file, "r", encoding='utf-8')
                save_dict = json.load(f)
                f.close()
                for type_ in save_dict:
                    for checkpoint in save_dict[type_]:
                        if len(save_dict[type_][checkpoint]["labels"]) > 1:
                            labels = save_dict[type_][checkpoint]["labels"][-1]
                            save_dict[type_][checkpoint]["labels"] = [labels]
        
        for root, dirs, files in os.walk(in_dir):
            for file_name in files:
                if not file_name.endswith("output_data.json"):
                    continue
                out_json = os.path.join(root, file_name)

                try:
                    save_dict = rm_patrolai(out_json, save_dict, out_dir)
                except:
                    print(out_json, "is wrong !!")
        
        f = open(save_file, "w", encoding='utf-8')
        json.dump(save_dict, f, ensure_ascii=False, indent=2)
        f.close()

        time.sleep(36000)

def reg_crop(img_ref,xmin,ymin,xmax,ymax):
    h, w = img_ref.shape[:2]
    xmin=int(float(xmin) * w)
    ymin=int(float(ymin) * h)
    xmax=int(float(xmax) * w)
    ymax=int(float(ymax) * h)
    img_empty = np.zeros((h, w, 3), np.uint8)
    img_empty[ymin:ymax, xmin:xmax] = img_ref[ymin:ymax, xmin:xmax]
    return img_empty


def img_fill(img, size,x1, y1, x2, y2 ):
    x_crop = x2 - x1
    y_crop = y2 - y1
    img = img[y1:y2, x1:x2]
    if x_crop > 450:
        img = cv2.resize(img, (450, int(y_crop * 450 / x_crop)))
        y_crop = int(y_crop * 450 / x_crop)
        x_crop = 450
    if y_crop > 450:
        img = cv2.resize(img, (int(x_crop * 450 / y_crop), 450))
        x_crop = int(x_crop * 450 / y_crop)
        y_crop = 450
        # print(x_crop)

    img_empty = np.zeros((size, size, 3), np.uint8)
    img_empty[size // 5:size // 5 + y_crop, size // 5:size // 5 + x_crop] += img
    return img_empty


def dp_append(label, dp):
    if dp >= len(label):
        label.insert(0, "0.")
    else:
        label.insert(len(label) - dp, ".")
    return label


#osd区域置灰
def osd_crop(img,coor):
    img[coor[1]:coor[3],coor[0]:coor[2]]=125
    return img

if __name__ == '__main__':
    rm_result_patrolai()
