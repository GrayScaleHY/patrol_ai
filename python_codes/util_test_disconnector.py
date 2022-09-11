import os
import cv2
import glob
from util_yjsk import json2bboxes, disconnector_state
import time

video_dir = "/data/yh/try_test/yjsk"
cfg_dir = "/data/yh/try_test/cfg"
result_dir = "/data/yh/try_test/result"

# video_list = glob.glob(os.path.join(video_dir, "*.mp4"))
video_list = [
    "/data/yh/try_test/yjsk/0035_normal.MP4"
]
for tag_video in video_list:
# tag_video = ""/data/yh/try_test/yjsk/0021_normal.mp4","
    print(tag_video)
    loop_start = time.time()

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
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps =cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(os.path.join(result_dir, video_name),fourcc, fps, size)

    while(cap.isOpened()):
        ret, img_tag = cap.read() # 逐帧读取

        if ret==True:

            state, scores, bboxes_tag = disconnector_state(img_tag, img_opens, img_closes, box_state, box_osd, img_yichangs)
            if state == "异常":
                color = (0, 0, 255)
                s = "yichang"
            elif state == "分":
                color = (255, 0, 0)
                s = "fen"
            else:
                s = "he"
                color = (0, 255, 0)

            cv2.putText(img_tag, s, (int(size[0]/2) - 100, 100),cv2.FONT_HERSHEY_SIMPLEX, 3, color, thickness=5)
            for i in range(2):
                b = bboxes_tag[i]
                s = str(scores[i])
                cv2.rectangle(img_tag, (int(b[0]), int(b[1])),(int(b[2]), int(b[3])), color, thickness=1)
                cv2.putText(img_tag, s, (int(b[0])-150, int(b[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)

            if len(bboxes) > 2:
                b = bboxes[2]
                cv2.rectangle(img_tag, (int(b[0]), int(b[1])),(int(b[2]), int(b[3])), color, thickness=2)

            out.write(img_tag)
            # cv2.imshow('frame',img_tag)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break

    ## 释放内存
    cap.release()
    out.release()

    print(f"loop time = {time.time() - loop_start}")
