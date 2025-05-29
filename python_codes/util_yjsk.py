import cv2
import numpy as np
import os
import glob
import time
import argparse
from lib_inference_yolov8 import inference_yolov8_classify, load_yolov8_model

yolov8_model = load_yolov8_model("/data/PatrolAi/yolov8/yjsk.pt")

def disconnector_state(real_model, img_tag):
    """:
    args:
        img_tag: 待分析图
    return: 
        state: 返回待分析图的当前状态,返回状态之一：无法判别状态、异常、分、 合]
        score: 分类得分
    """
    cfgs = inference_yolov8_classify(real_model, img_tag,resize=224)

    label = cfgs[0]["label"]; score = cfgs[0]["score"]

    ## 判断当前刀闸状态
    if label == "fen":
        state = "分"
    elif label == "he":
        state = "合"
    else:
        state = "异常"

    return state, score

def video_states(tag_video):
    """
    获取tag_video的状态列表。
    args:
        tag_video: 待分析视频
    return:
        states_start: 视频状态列表，例如：[分, 异常, 合]
        states_end: 视频状态列表，例如：[分, 异常, 合]
    """

    step = 1 # 多少帧抽一帧
    context = 10 # 看开头或结尾几帧

    cap = cv2.VideoCapture(tag_video) ## 建立视频对象
    frame_number = cap.get(7)  # 视频文件的帧数
    if frame_number < context * 2:
        cap.release() # 释放内存
        print("Warning:", tag_video, "is wrong !")
        return ["分"] * context, ["异常"] * context
        
    ## 取帧，头几帧存放在start_frames中，尾几帧存放在end_frames中
    start_frames = []
    end_frames = [np.ones((3,3,3),dtype=np.uint8)] * context
    end_counts = [0] * context
    counts = []
    count = -1
    while(cap.isOpened()):
        ret, img_tag = cap.read() # 逐帧读取
        
        if ret==True and img_tag is not None:
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
            count += 1
        else:
            break
    cap.release() # 释放内存

    ## 推理生成states
    states_start = []
    states_end = []
    scores_start = []
    scores_end = []
    real_model = yolov8_model

    for img_tag in start_frames: 
        state, score = disconnector_state(real_model, img_tag)
        states_start.append(state)
        scores_start.append(score)
    
    for img_tag in end_frames:
        state, score = disconnector_state(real_model, img_tag)
        states_end.append(state)
        scores_end.append(score)

    counts = counts + end_counts
    print("frame indexs:", counts)
    print("states_start list:", states_start, "states_end list:", states_end)
    print("scores_start list:", scores_start, "scores_end list:", scores_end)

    return states_start, states_end, img_tag

def final_state(states_start, states_end, len_window=4):
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

    # if len(states_end) <= len_window or len(states_start) <= len_window:
    #     return 2

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
    if state_end == "合" and state_start == "分":
        print("1, 合闸正常")
        return 1 # 合闸正常
    elif state_end == "分" and state_start == "合":
        print("2, 分闸正常")
        return 2 # 分闸正常
    else: 
        if state_start == "分":
            print("3, 合闸异常")
            return 3 # 合闸异常
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

    # ## 保存图片的路径
    # save_copy_dir = "/home/data/yjsk"
    # os.makedirs(save_copy_dir, exist_ok=True)

    for tag_video in video_list:
        
        if "_normal." in tag_video:
            continue
        
        print("----------------------------------------")
        print(tag_video)
        start_loop = time.time()

        # ## 保存一份原图
        # save_copy_file = os.path.join(save_copy_dir, os.path.basename(tag_video))
        # if not os.path.exists(save_copy_file):
        #     shutil.copy(tag_video, save_copy_file)
        
        states_start, states_end = video_states(tag_video) # 求tag_video的状态列表
        f_state = final_state(states_start, states_end, len_window=4) # 求最终状态
        print(f_state)

        ## 保存比赛的格式
        tag_name = os.path.basename(tag_video)
        id_ = 1
        s = "ID,NAME,TYPE\n"
        s = s + str(id_) + "," + tag_name + "," + str(f_state) + "\n"
        f = open(os.path.join(out_dir, tag_name[:-4] + ".txt"), "w", encoding='utf-8')
        f.write(s)
        f.close()

        print("spend loop time:", time.time() - start_loop)
    print("spend time total:", time.time() - start_all)
