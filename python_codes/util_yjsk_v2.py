from lib_sift_match import my_ssim, sift_create, sift_match, cw_ssim_index
from lib_img_registration import convert_coor, correct_offset
import cv2
import json
import numpy as np
import os
import glob
import time
import argparse
from lib_inference_yolov5 import inference_yolov5, load_yolov5_model

yolov5_disconnector_2classes = load_yolov5_model("/mnt/data/PatrolAi/yolov5/disconnector_2classes.pt") # [分到位， 合到位]

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

def disconnector_state_yolov5(img_tag, l_type="end"):
    """
    使用yolov5模型对刀闸进行判别
    args:
        img_tag: 待分析图
        l_type: 起始位置或者结束位置, end or start
    return:
        state: 返回待分析图的当前状态, 状态包含"分", "合", "空"三种
        scores: 每个box里面的得分,[score1, score2, ..]
        bboxes_tag: 模板图上的两个框框映射到待分析图上的大概位置，[[xmin, ymin, xmax, ymax], ..]
    """
    if l_type == "end":
        conf_thres = 0.6
    else:
        conf_thres = 0.2
    try:
        cfgs = inference_yolov5(yolov5_disconnector_2classes, img_tag, conf_thres=conf_thres)
    except:
        return "异常", [], []

    if len(cfgs) == 0:
        return "异常", [], []
    
    ## 判断cfgs中最大得分的label是分还是合
    scores = []
    bboxes_tag = []
    score_max = 0
    for cfg in cfgs:
        score = cfg["score"]
        label = cfg["label"]
        if score > score_max:
            score_max = score
            if label == "分到位":
                state = "分"
            else:
                state = "合"
            
        scores.append(round(score, 2))
        bboxes_tag.append(cfg["coor"])
    
    return state, scores, bboxes_tag

def disconnector_state(img_tag, img_opens, img_closes, box_state, box_osd=[], img_yichangs=[]):
    """:
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

    feat_open = sift_create(img_opens[0], rm_regs=box_osd[2:]) # 提取参考图sift特征
    feat_tag = sift_create(img_tag, rm_regs=box_osd[2:]) # 提取待分析图sift特征

    img_tag_ = img_tag.copy() 

    M = sift_match(feat_open, feat_tag, ratio=0.5, ops="Affine") # 求偏移矩阵
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
        if 0 <= bbox[0] <= bbox[2] <= W and 0 <= bbox[1] <= bbox[3] <= H:
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
        scores.append([round(score_close, 2), round(score_open, 2), round(score_yc, 2)])
    
    ## 判断当前刀闸状态
    if all(state_ == "open" for state_ in states):
        state = "分"
    elif all(state_ == "close" for state_ in states):
        state = "合"
    else:
        state = "异常"

    return state, scores, bboxes_tag

def video_states(tag_video, cfg_dir, use_model=True):
    """
    获取tag_video的状态列表。
    args:
        tag_video: 待分析视频
        img_open: 刀闸分模板图
        img_close: 刀闸合模板图
        bboxes: 配置的框子信息
    return:
        states_start: 视频状态列表，例如：[分, 异常, 合]
        states_end: 视频状态列表，例如：[分, 异常, 合]
    """
    video_name = os.path.basename(tag_video)
    v_id = video_name.split("_")[0]

    json_file = os.path.join(cfg_dir, v_id + "_normal.json")
    open_files = [os.path.join(cfg_dir, v_id + "_normal_off.png")]
    close_files = [os.path.join(cfg_dir, v_id + "_normal_on.png")]
    yc_files = glob.glob(os.path.join(cfg_dir, v_id + "_normal_*yc.png"))
    img_opens = [cv2.imread(f_) for f_ in open_files]
    img_closes = [cv2.imread(f_) for f_ in close_files]
    img_yichangs = [cv2.imread(f_) for f_ in yc_files]
    bboxes = json2bboxes(json_file, img_opens[0])
    box_state = bboxes[:2]
    box_osd = bboxes[2:]

    step = 2 # 多少帧抽一帧
    context = 7 # 看开头或结尾几帧

    cap = cv2.VideoCapture(tag_video) ## 建立视频对象
    frame_number = cap.get(7)  # 视频文件的帧数
    if frame_number < 12:
        cap.release() # 释放内存
        print("Warning:", tag_video, "is wrong !")
        return ["分"] * 7, ["异常"] * 7
        
    states_start = []
    states_end = []
    scores_start = []
    scores_end = []
    counts = []
    count = -1

    start_frames = []
    end_frames = [np.ones((3,3,3),dtype=np.uint8)] * context
    end_counts = [0] * context

    while(cap.isOpened()):
        ret, img_tag = cap.read() # 逐帧读取
        

        if ret==True and img_tag is not None:
            count += 1
            if np.sum(img_tag) < 100:
                continue
            
            if count % step == 0 and len(start_frames) < context:
                start_frames.append(img_tag)
                counts.append(count)

            if count % step == 0:
                end_frames.pop(0)
                end_frames.append(img_tag)
                end_counts.pop(0)
                end_counts.append(count)
     
        else:
            break

    for img_tag in start_frames: # 抽前10帧和后10帧
        if use_model:
            state, scores, _ = disconnector_state_yolov5(img_tag, l_type="start")
            if state == "异常":
                state, scores, _ = disconnector_state(img_tag, img_opens, img_closes, box_state, box_osd, img_yichangs)
        else:
            state, scores, _ = disconnector_state(img_tag, img_opens, img_closes, box_state, box_osd, img_yichangs)
        states_start.append(state)
        scores_start.append(scores)
        counts.append(count)
    
    for img_tag in end_frames:
        if use_model:
            state, scores, _ = disconnector_state(img_tag, img_opens, img_closes, box_state, box_osd, img_yichangs)
            if state == "合" or state == "分":
                state, scores, _ = disconnector_state_yolov5(img_tag, l_type="end")
        else:
            state, scores, _ = disconnector_state(img_tag, img_opens, img_closes, box_state, box_osd, img_yichangs)
        states_end.append(state)
        scores_end.append(scores)

    counts = counts + end_counts
    print("frame indexs:", counts)
    print("states_start list:", states_start, "states_end list:", states_end)
    print("scores_start list:", scores_start, "scores_end list:", scores_end)

    cap.release() # 释放内存

    return states_start, states_end

def final_state(states_start, states_end, len_window=3):
    """
    判断states列表的动状态。
    用固定大小的滑窗在states上滑动，当滑窗内的元素都相同，则表示为该时刻的状态，通过对比起始状态和结尾状态判断states的动状态。
    args:
        states_start: 视频起始状态列表，例如["分","分","分","异常","异常","异常","合","合","合"]
        states_end: 视频结束状态列表，例如["分","分","分","异常","异常","异常","合","合","合"]
        len_window: 滑窗大小
    return:
        0 代表无法判别状态; 1 代表合闸正常; 2 代表合闸异常; 3 代表分闸正常; 4 代表分闸异常。
    """

    if len(states_end) <= len_window or len(states_start) <= len_window:
        return 2

    ## 根据states_end和states_start判断出视频起始状态和最终状态。
    is_yc = True
    for i in range(len(states_end) - len_window + 1):
        s_types = set(states_end[i:i+len_window])
        if s_types == {"分"} or s_types == {"合"}:
            state_end = list(s_types)[0]
            is_yc = False
    if is_yc:
        state_end = "异常"
    
    start_count = [states_start.count(i) for i in ["分", "合", "异常", "无法识别状态"]]
    if start_count[0] > start_count[1]:
        state_start = "分"
    else:
        state_start = "合"

    ## 根据state_start合state_end的组合判断该states的最终状态
    ## 其中 0 代表无法判别状态，1 代表合闸正常， 2 代表合闸异常，3 代表分闸正常，4 代表分闸异常
    if state_end == "合":
        print("1, 合闸正常")
        return 1 # 合闸正常
    elif state_end == "分":
        print("3, 分闸正常")
        return 3 # 分闸正常
    else: 
        if state_start == "分":
            print("2, 合闸异常")
            return 2 # 合闸异常
        elif state_start == "合":
            print("4, 分闸异常")
            return 4 # 分闸异常
        else:
            print("2, 合闸异常")
            return 2 # 无法判别状态
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source',
        type=str,
        default='./test/yjsk',
        help='source dir.')
    parser.add_argument(
        '--out_dir',
        type=str,
        default='./result/yjsk40zhytdlkjgfyxgs',
        help='out dir of saved result.')
    parser.add_argument(
        '--cfgs',
        type=str,
        default='./cfgs',
        help='cfg dir')
    parser.add_argument(
        '--data_part',
        type=str,
        default='1/1',
        help='part of data split.')
    args, unparsed = parser.parse_known_args()

    # video_dir = "test/yjsk"
    # cfg_dir = "test/cfg"
    # out_dir = "test/tuilishuju_output"
    video_dir = args.source # 待测试文件目录
    out_dir = args.out_dir # 结果保存目录
    cfg_dir = args.cfgs # md5列表目录
    data_part = args.data_part # 分隔数据部分

    os.makedirs(out_dir, exist_ok=True)
    start_all = time.time()

    ## 分割数据
    video_list = glob.glob(os.path.join(video_dir, "*"))
    video_list.sort()
    _s = int(data_part.split("/")[1])
    _p = int(data_part.split("/")[0])
    _l = len(video_list)
    if _s != _p:
        video_list = video_list[int(_l*(_p-1)/_s):int(_l*_p/_s)]
    else:
        video_list = video_list[int(_l*(_p-1)/_s):]

    for tag_video in video_list:
        
        if "_normal." in tag_video:
            continue
        
        print("----------------------------------------")
        print(tag_video)
        start_loop = time.time()
        
        ## 判断是否有模型
        if os.path.basename(tag_video).startswith("0040_"):
            use_model = False
        else:
            use_model = True

        states_start, states_end = video_states(tag_video, cfg_dir, use_model=use_model) # 求tag_video的状态列表
        f_state = final_state(states_start, states_end, len_window=3) # 求最终状态
        print(f_state)

        ## 保存比赛的格式
        tag_name = os.path.basename(tag_video)
        id_ = 1
        s = "ID,Name,Type\n"
        s = s + str(id_) + "," + tag_name + "," + str(f_state) + "\n"
        f = open(os.path.join(out_dir, tag_name[:-4] + ".txt"), "w", encoding='utf-8')
        f.write(s)
        f.close()

        print("spend loop time:", time.time() - start_loop)
    print("spend time total:", time.time() - start_all)
