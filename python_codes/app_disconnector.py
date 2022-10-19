
from lib_image_ops import base642img, img2base64, img_chinese
from lib_sift_match import sift_create, sift_match, correct_offset, convert_coor, cw_ssim_index
import time
import json
import cv2
import numpy as np
import os

state_map = {
    "合": {"name": "合闸正常", "color": [(255,0,0), (255,0,0)]},
    "分": {"name": "分闸正常", "color": [(0,255,0), (0,255,0)]},
    "异常": {"name": "分合闸异常", "color": [(0,0,255), (0,0,255)]},
    "无法判别状态": {"name": "分析失败", "color": [(0,0,255), (0,0,255)]},
}

def json2bboxes(json_file, img_open):
    """
    将json_file里的框子信息提取出来。
    return:
        bboxes: 格式，[[xmin, ymin, xmax, ymax], ..]
    """
    f = open(json_file, "r", encoding='utf-8')
    cfg = json.load(f)
    f.close()
    H, W = img_open.shape[:2]
    ids = [id_ for id_ in cfg]
    ids.sort()
    bboxes = []
    for id_ in ids:
        c = np.array([[cfg[id_][0]["x"], cfg[id_][0]["y"]],[cfg[id_][1]["x"], cfg[id_][1]["y"]]])
        coor = [np.min(c[:,0]), np.min(c[:,1]), np.max(c[:,0]), np.max(c[:,1])]
        coor = np.array(coor) * np.array([W, H, W, H])
        bboxes.append(coor.astype(int).tolist())
    return bboxes


def disconnector_state(img_tag, img_opens, img_closes, box_state, box_osd=[], img_yichangs=[]):
    """
    刀闸分合判别， 可支持多模板以及异常图片模板。
    args:
        img_tag: 待分析图
        img_opens: 分闸模板图, list
        img_closes: 合闸模板图, list
        box_state: 用于对比ssim的框子坐标，格式为[[xmin, ymin, xmax, ymax], ..]
        box_osd: sift匹配时需要扣掉的区域，格式为[[xmin, ymin, xmax, ymax], ..]
        img_yichangs: 异常模板图,剪切过后的, list
    return: 
        state: 返回待分析图的当前状态,返回状态之一：无法判别状态、异常、分、 合]
        scores: 每个box里面的得分,[[score_close, score_open, score_yc], ..]
        bboxes_tag: 模板图上的两个框框映射到待分析图上的大概位置，[[xmin, ymin, xmax, ymax], ..]
    """
    assert isinstance(img_opens, list) and len(img_opens) > 0, "img_opens is not requested !"
    assert isinstance(img_closes, list) and len(img_closes) > 0, "img_closes is not requested !"
    assert isinstance(box_state, list) and len(box_state) > 0, "box_state is not requested !"

    feat_open = sift_create(img_opens[0]) # 提取参考图sift特征
    feat_tag = sift_create(img_tag) # 提取待分析图sift特征

    img_tag_ = img_tag.copy() 

    M = sift_match(feat_open, feat_tag, rm_regs=box_osd[2:], ratio=0.5, ops="Affine") # 求偏移矩阵
    img_tag_warped = correct_offset(img_tag, M) # 对待分析图进行矫正

    
    ## 将框子区域在open和close和tag文件中画出来，以方便查看矫正偏移对不对
    # open_ = img_open.copy()
    # close_ = img_close.copy()
    # tag_ = img_tag_warped.copy()
    img_open= img_opens[0]

    ## 将模板图上的bbox映射到待分析图上，求bboxes_tag
    bboxes_tag = []
    for b in box_state:
        
        bbox = list(convert_coor((b[0], b[1]), M)) + list(convert_coor((b[2], b[3]), M))
        bboxes_tag.append(bbox)

    ## 分析box_state里面的刀闸状态
    states = []
    s = ""
    scores = []
    for bbox in box_state:
        ## 判断bbox是否符合要求
        H, W = img_tag.shape[:2]
        if 0 < bbox[0] < bbox[2] < W and 0 < bbox[1] < bbox[3] < H:
            pass
        else:
            return "无法判别状态", scores, bboxes_tag

        ## 截取图片区域，并且用ssim算法比较相似性
        img_ta = img_tag_warped[bbox[1]: bbox[3], bbox[0]: bbox[2]]

        ## 求score open
        score_open = 0
        for img_open in img_opens:
            img_op = img_open[bbox[1]: bbox[3], bbox[0]: bbox[2]]
            score = cw_ssim_index(img_ta, img_op)
            if score > score_open:
                score_open = score

        ## 求score close
        score_close = 0
        for img_close in img_closes:
            img_cl = img_close[bbox[1]: bbox[3], bbox[0]: bbox[2]]
            score = cw_ssim_index(img_ta, img_cl)
            if score > score_close:
                score_close = score

        ## 求score yichang
        score_yc = 0
        for img_yc in img_yichangs:
            if img_yc.shape != img_ta.shape:
                continue
            score = cw_ssim_index(img_ta, img_yc)
            if score > score_yc:
                score_yc = score

        ## 根据score_open和score_close对比，判断该框框的状态。
        if score_yc > score_open and score_yc > score_close:
            state_ = "yichang"
        elif score_close > score_open:
            state_ = "close"
        else:
            state_ = "open"
        states.append(state_)
        scores.append([round(score_close, 3), round(score_open, 3), round(score_yc, 3)])
    
    ## 判断当前刀闸状态
    if all(state_ == "open" for state_ in states):
        state = "分"
    elif all(state_ == "close" for state_ in states):
        state = "合"
    else:
        state = "异常"

    return state, scores, bboxes_tag

def get_input_data(input_data):
    """
    提取input_data中的信息。
    return:
        img_tag: 目标图片数据
        img_open: 模板图，刀闸打开
        img_close: 模板图，刀闸闭合
        roi1, roi2: 感兴趣区域, 结构为[xmin, ymin, xmax, ymax]
    """

    img_tag = base642img(input_data["image"])

    ## 是否有模板图
    img_close = None
    if "img_close" in input_data["config"]:
        if isinstance(input_data["config"]["img_close"], str):
            img_close = base642img(input_data["config"]["img_close"])    

    ## 是否有模板图
    img_open = None
    if "img_open" in input_data["config"]:
        if isinstance(input_data["config"]["img_open"], str):
            img_open = base642img(input_data["config"]["img_open"]) 
        
    ## 感兴趣区域
    roi1= None # 初始假设
    roi2= None # 初始假设
    if "bboxes" in input_data["config"]:
        if isinstance(input_data["config"]["bboxes"], dict):
            if "roi1" in input_data["config"]["bboxes"]:
                if isinstance(input_data["config"]["bboxes"]["roi1"], list):
                    if len(input_data["config"]["bboxes"]["roi1"]) == 4:
                        W = img_open.shape[1]; H = img_open.shape[0]
                        roi1 = input_data["config"]["bboxes"]["roi1"]
                        roi1 = [int(roi1[0]*W), int(roi1[1]*H), int(roi1[2]*W), int(roi1[3]*H)]
            if "roi1" in input_data["config"]["bboxes"]:
                if isinstance(input_data["config"]["bboxes"]["roi2"], list):
                    if len(input_data["config"]["bboxes"]["roi2"]) == 4:
                        W = img_open.shape[1]; H = img_open.shape[0]
                        roi2 = input_data["config"]["bboxes"]["roi2"]
                        roi2 = [int(roi2[0]*W), int(roi2[1]*H), int(roi2[2]*W), int(roi2[3]*H)]
    
    return img_tag, img_open, img_close, roi1, roi2

def inspection_disconnector(input_data):
    """
    刀闸识别
    """
    TIME_START = time.strftime("%m-%d-%H-%M-%S") + "_"
    save_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_path = os.path.join(save_path, "result_patrol", input_data["type"])
    os.makedirs(save_path, exist_ok=True)
    f = open(os.path.join(save_path, TIME_START + "input_data.json"), "w")
    json.dump(input_data, f, ensure_ascii=False)  # 保存输入信息json文件
    f.close()
    
    ## 提取data信息
    out_data = {"code": 0, "data":{}, "msg": "Success request disconnector"}
    img_tag, img_open, img_close, roi1, roi2 = get_input_data(input_data)

    ## 保存模板图与待分析图
    cv2.imwrite(os.path.join(save_path, TIME_START + "img_close.jpg"), img_close)
    cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag.jpg"), img_tag)
    cv2.imwrite(os.path.join(save_path, TIME_START + "img_open.jpg"), img_open)

    ## 求刀闸状态
    img_opens = [img_open]
    img_closes = [img_close]
    bboxes = [roi1, roi2]
    state, _, bboxes_tag = disconnector_state(img_tag, img_opens, img_closes, bboxes, box_osd=[], img_yichangs=[])

    out_data["data"] = {"result": state_map[state]["name"]}

    ## 保存结果
    for i in range(2):
        b1 = bboxes[0]; b2 = bboxes[1]
        bt1 = bboxes_tag[0]; bt2 = bboxes_tag[1]
        cv2.rectangle(img_open, (b1[0], b1[1]), (b1[2], b1[3]), state_map["分"]["color"][0], thickness=2)
        cv2.rectangle(img_open, (b2[0], b2[1]), (b2[2], b2[3]), state_map["分"]["color"][0], thickness=2)
        cv2.rectangle(img_close, (b1[0], b1[1]), (b1[2], b1[3]), state_map["合"]["color"][0], thickness=2)
        cv2.rectangle(img_close, (b2[0], b2[1]), (b2[2], b2[3]), state_map["合"]["color"][0], thickness=2)
        cv2.rectangle(img_tag, (bt1[0], bt1[1]), (bt1[2], bt1[3]), state_map[state]["color"][0], thickness=2)
        cv2.rectangle(img_tag, (bt2[0], bt2[1]), (bt2[2], bt2[3]), state_map[state]["color"][1], thickness=2)
    img_tag = img_chinese(img_tag, state_map[state]["name"], (10, 50), color=state_map[state]["color"][0], size=40)

    cv2.imwrite(os.path.join(save_path, TIME_START + "img_open_cfg.jpg"), img_open)
    cv2.imwrite(os.path.join(save_path, TIME_START + "img_close_cfg.jpg"), img_close)
    cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag_cfg.jpg"), img_tag)

    f = open(os.path.join(save_path, TIME_START + "output_data.json"),"w",encoding='utf-8')
    json.dump(out_data, f, indent=2)
    f.close()

    out_data["img_result"] = img2base64(img_tag)

    return out_data

if __name__ == '__main__':
    tag_file = "/home/yh/image/python_codes/test/test1/img_tag2.jpg"
    open_file = "/home/yh/image/python_codes/test/test1/img_open.jpg"
    close_file = "/home/yh/image/python_codes/test/test1/img_close.jpg"
    
    bboxes = [[651, 315, 706, 374], [661, 400, 713, 450]]
    img_close = img2base64(cv2.imread(close_file))
    img_open = img2base64(cv2.imread(open_file))
    img_tag = img2base64(cv2.imread(tag_file))
    H, W = cv2.imread(close_file).shape[:2]
    roi1 = [bboxes[0][0] / W, bboxes[0][1] / H, bboxes[0][2] / W, bboxes[0][3] / H]
    roi2 =  [bboxes[1][0] / W, bboxes[1][1] / H, bboxes[1][2] / W, bboxes[1][3] / H]
    input_data = {
        "image": img_tag,
        "config": {"img_open": img_open, "img_close": img_close, "bboxes": {"roi1": roi1, "roi2": roi2}},
        "type": "disconnector"
    }
    out_data = inspection_disconnector(input_data)
    for s in out_data:
        if s != "img_result":
            print(s,":",out_data[s])

