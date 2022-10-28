import os
import cv2
import glob
import time
import numpy as np
from util_yjsk_v2 import json2bboxes, disconnector_state, disconnector_state_yolov5, yolov5_disconnector_2classes

video_dir = "/data/yh/try_test/yjsk"
cfg_dir = "/data/yh/try_test/cfg"
result_dir = "/data/yh/try_test/result"

video_list = glob.glob(os.path.join(video_dir, "*.*"))
# video_list = [
#     "/data/yh/try_test/yjsk/0035_normal.MP4"
# ]
for tag_video in video_list:
# tag_video = "/data/yh/try_test/yjsk/0021_normal.mp4"
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

    cap = cv2.VideoCapture(tag_video)
    frame_number = cap.get(7)  # 视频文件的帧数
    print(frame_number)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps =cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(os.path.join(result_dir, video_name),fourcc, fps, size)

    step = 1 # 多少帧抽一帧
    context = 28 # 看开头或结尾几帧
    states_start = []
    states_end = []
    scores_start = []
    scores_end = []
    counts = []
    count = 0
    while(cap.isOpened()):
        ret, img_tag = cap.read() # 逐帧读取
        

        if ret==True:
            if np.sum(img_tag) < 100:
                continue
            is_write = False
            if count < context * step : # 抽前10帧和后10帧
                is_write = True

                if use_model:
                    state, scores,bboxes_tag= disconnector_state_yolov5(img_tag, l_type="start")
                    if state == "异常":
                        state, scores,bboxes_tag= disconnector_state(img_tag, img_opens, img_closes, box_state, box_osd, img_yichangs)
                else:
                    state, scores,bboxes_tag= disconnector_state(img_tag, img_opens, img_closes, box_state, box_osd, img_yichangs)
                states_start.append(state)
                scores_start.append(scores)
                counts.append(count)
            
            elif count >= frame_number - context * step:
                is_write = True
                if use_model:
                    state, scores,bboxes_tag= disconnector_state(img_tag, img_opens, img_closes, box_state, box_osd, img_yichangs)
                    if state == "合" or state == "分":
                        state, scores,bboxes_tag= disconnector_state_yolov5(img_tag, l_type="end")
                else:
                    state, scores,bboxes_tag= disconnector_state(img_tag, img_opens, img_closes, box_state, box_osd, img_yichangs)
                states_end.append(state)
                scores_end.append(scores)
                counts.append(count)

            # state, scores, bboxes_tag, labels = disconnector_state_yolov5(yolov5_disconnector_2classes, img_tag)
            if is_write:
                if state == "异常":
                    color = (0, 0, 255)
                    s = "yichang"
                elif state == "分":
                    color = (255, 0, 0)
                    s = "fen"
                else:
                    s = "he"
                    color = (0, 255, 0)

                cv2.putText(img_tag, s, (int(size[0]/2) - 100, 200),cv2.FONT_HERSHEY_SIMPLEX, 3, color, thickness=5)
                for i in range(len(bboxes_tag)):
                    b = bboxes_tag[i]
                    s = str(scores[i])
                    cv2.rectangle(img_tag, (int(b[0]), int(b[1])),(int(b[2]), int(b[3])), color, thickness=1)
                    cv2.putText(img_tag, s, (int(b[0]), int(b[1])+50),cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)

            count += 1
            out.write(img_tag)
            # cv2.imshow('frame',img_tag)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break

    ## 释放内存
    cap.release()
    out.release()

    print(f"loop time = {time.time() - start_loop}")
