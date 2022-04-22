from lib_sift_match import my_ssim, sift_create, sift_match, correct_offset, convert_coor
import cv2
import json
import numpy as np
import os
import glob


def json2bboxes(json_file):
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


def disconnector_state(img_open, img_close, img_tag, bboxes):
    """
    刀闸分合判别
    args:
        img_open: 分闸模板图
        img_close: 合闸模板图
        img_tag: 待分析图
        bboxes: 框子信息,通常，前两个bbox表示ssim对比区域，后几个bbox表示OSD区域，
                格式为[[xmin, ymin, xmax, ymax], ..]
    return: 
        state: 返回待分析图的当前状态,返回状态之一：无法判别状态、异常、分、 合]
        bboxes_tag: 模板图上的两个框框映射到待分析图上的大概位置，[[xmin, ymin, xmax, ymax], ..]
    """
    feat_open = sift_create(img_open) # 提取参考图sift特征
    feat_tag = sift_create(img_tag) # 提取待分析图sift特征

    M = sift_match(feat_open, feat_tag, rm_regs=bboxes[2:]) # 求偏移矩阵
    img_tag_warped = correct_offset(img_tag, M) # 对待分析图进行矫正

    ## 将模板图上的bbox映射到待分析图上，求bboxes_tag
    bboxes_tag = []
    for b in bboxes[:2]:
        bbox = list(convert_coor((b[0], b[1]), M)) + list(convert_coor((b[2], b[3]), M))
        bboxes_tag.append(bbox)

    ## 分析两个bbox里面的刀闸状态
    states = []
    for bbox in bboxes[:2]:
        ## 判断bbox是否符合要求
        H, W = img_tag.shape[:2]
        if 0 < bbox[0] < bbox[2] < W and 0 < bbox[1] < bbox[3] < H:
            pass
        else:
            return "无法判别状态", bboxes_tag

        ## 截取图片区域，并且用ssim算法比较相似性
        img_op = img_open[bbox[1]: bbox[3], bbox[0]: bbox[2]]
        img_cl = img_close[bbox[1]: bbox[3], bbox[0]: bbox[2]]
        img_tag_warp = img_tag_warped[bbox[1]: bbox[3], bbox[0]: bbox[2]]
        score_open = my_ssim(img_tag_warp, img_op) #计算ssim结构相似性
        score_close = my_ssim(img_tag_warp, img_cl)

        ## 根据score_open和score_close对比，判断该框框的状态。
        if score_close > score_open:
            states.append("close")
        else:
            states.append("open")
    
    ## 判断当前刀闸状态
    if states[0] != states[1]:
        return "异常", bboxes_tag
    
    if states[0] == "open" and states[1] == "open":
        return "分", bboxes_tag

    if states[0] == "close" and states[1] == "close":
        return "合", bboxes_tag


def video_states(tag_video, img_open, img_close, bboxes):
    """
    获取tag_video的状态列表。
    args:
        tag_video: 待分析视频
        img_open: 刀闸分模板图
        img_close: 刀闸合模板图
        bboxes: 配置的框子信息
    return:
        states: 视频状态列表，例如：[分, 异常, 合]
    """
    cap = cv2.VideoCapture(tag_video) ## 建立视频对象
    states = []
    step = 3
    count = 0
    while(cap.isOpened()):

        ret, img_tag = cap.read() # 逐帧读取

        if ret==True:
            if count % step == 0:
                state, _ = disconnector_state(img_open, img_close, img_tag, bboxes)
                states.append(state)
            count += 1
        else:
            break

    cap.release() # 释放内存
    return states

def final_state(states, len_window=5):
    """
    判断states列表的动状态。
    用固定大小的滑窗在states上滑动，当滑窗内的元素都相同，则表示为该时刻的状态，通过对比起始状态和结尾状态判断states的动状态。
    args:
        states: 状态列表，例如["分","分","分","异常","异常","异常","合","合","合"]
        len_window: 滑窗大小
    return:
        0 代表无法判别状态; 1 代表合闸正常; 2 代表合闸异常; 3 代表分闸正常; 4 代表分闸异常。
    """

    ## 用len_window长度的滑窗滑动states, 得出state_start和state_end
    state_start = ""
    state_end = ""
    for i in range(len(states) - (len_window - 1)):
        s_types = set(states[i:i+len_window]) # 取len_window长度的元素，并去除不同元素
        if s_types == {"分"} or s_types == {"合"}:
            state_end = list(s_types)[0]
            if state_start == "":
                state_start = list(s_types)[0]

    ## 根据state_start合state_end的组合判断该states的最终状态
    if state_start == "分" and state_end == "合":
        return 1 # 合闸正常
    if state_start == "分" and state_end != "合":
        return 2 # 合闸异常
    if state_start == "合" and state_end == "分":
        return 3 # 分闸正常
    if state_start == "合" and state_end != "分":
        return 4 # 分闸异常
    else:
        return 0 # 无法判别状态
    

if __name__ == "__main__":

    video_dir = "test/yjsk"
    cfg_dir = "test/cfg"
    out_dir = "test/tuilishuju_output"
    os.makedirs(out_dir, exist_ok=True)

    for ref_video in glob.glob(os.path.join(video_dir, "*_normal.mp4")):

        file_id = os.path.basename(ref_video).split("_")[0]

        open_file = os.path.join(cfg_dir, file_id + "_normal_off.jpg")
        close_file = os.path.join(cfg_dir, file_id + "_normal_on.jpg")
        json_file = os.path.join(cfg_dir, file_id + "_normal.json")

        img_open = cv2.imread(open_file)
        img_close = cv2.imread(close_file)
        bboxes = json2bboxes(json_file) # 获取json文件中的bboxes信息
        
        for tag_video in glob.glob(os.path.join(video_dir, file_id + "_*.mp4")):

            if tag_video.endswith("normal.mp4"):
                continue

            states = video_states(tag_video, img_open, img_close, bboxes) # 求tag_video的状态列表
            print("state list:", states)
            f_state = final_state(states, len_window=5) # 求最终状态
            print("final state:", f_state)

            ## 保存比赛的格式
            tag_name = os.path.basename(tag_video)
            id_ = int(tag_name.split("_")[-1][:-4])
            s = "ID,Name,Type\n"
            s = s + str(id_) + "," + tag_name + "," + str(f_state)
            f = open(os.path.join(out_dir, tag_name[:-4] + ".txt"), "w", encoding='utf-8')
            f.write(s)
            f.close()