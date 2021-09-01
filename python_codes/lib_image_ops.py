# -*- coding: UTF-8 -*-

import os
import glob
import cv2  # conda install opencv || pip install opencv-python
import ffmpeg  # pip install ffmpeg-python
import shutil
from PIL import Image
from moviepy.editor import VideoFileClip, concatenate_videoclips
import random
import base64
import numpy as np


def bmp2jpg(bmp_file, jpg_file):
    """
    bmp图片转jpg图片
    """
    img_data = cv2.imread(bmp_file, -1)  # 读取bmp图片
    cv2.imwrite(jpg_file, img_data)


def col2gray(col_file, gray_file):
    """
    彩图转灰度图
    """
    img_data = cv2.imread(col_file, 1)
    cv2.imwrite(gray_file, img_data)


def IsValidImage(img_file):
    """
    判断文件是否为有效（完整）的图片
    """
    bValid = True
    try:
        Image.open(img_file).verify()
    except:
        bValid = False
    return bValid


def img2jpg(img_file, out_file):
    """
    将图片转换为jpg格式。
    """
    assert IsValidImage(img_file), img_file + " 不是有效（完整）的图片"
    try:
        im = Image.open(img_file)
        im.save(out_file)
    except:
        print("warning:", img_file, "connot convert jpg_file ！")


def get_video_info(video_file):
    """
    获取视频的信息。
    args: 
        video_file: 视频文件路径
    return:
        format_name：视频格式
        codec_name： 编码格式
        duration_ts: 视频时长
        fps: 帧率
        width : 宽
        height: 长
    """
    info = ffmpeg.probe(video_file)
    vs = next(c for c in info['streams'] if c['codec_type'] == 'video')

    format_name = info['format']['format_name']
    codec_name = vs['codec_name']
    duration_ts = float(vs['duration_ts'])
    fps = vs['r_frame_rate']
    width = vs['width']
    height = vs['height']

    print("format_name:{} \ncodec_name:{} \nduration_ts:{} \nwidth:{} \nheight:{} \nfps:{}".format(
        format_name, codec_name, duration_ts, width, height, fps))

    return format_name, codec_name, duration_ts, fps, width, height


def clip_video(input_file, output_file, start, end):
    """
    截取或合并MP4格式的视频。
    """
    # 读取视频，并截取指定时间段的数据
    clip = VideoFileClip(input_file).subclip(start, end)
    # 视频合并
    # final_clip = concatenate_videoclips([clip1, clip3, clip2])
    # 视频写入mp4格式视频
    clip.write_videofile(output_file)


def video2imgs(video_file, img_dir, stride):
    """
    从视频中隔帧取图片。
    """
    print("precessing", video_file)
    os.makedirs(img_dir, exist_ok=True)
    video_name = os.path.basename(video_file)
    videoCapture = cv2.VideoCapture(video_file)
    # videoCapture=cv2.VideoCapture(1) # 通过摄像头的方式
    do_ = True
    f_count = 0
    while do_:
        do_, img_data = videoCapture.read()  # 逐帧读图片
        if f_count % stride == 0:
            img_file = os.path.join(
                img_dir, video_name[:-4] + "_" + str(f_count) + ".jpg")
            print(">>>", img_file)
            cv2.imwrite(img_file, img_data)
        f_count += 1
    videoCapture.release()


def draw_bboxs(img_file, bbox_cfg, is_write=False, is_show=False):
    """
    在图片上画bbox框
    args:
        img_file: 图片路径
        bbox_cfg: json文件，格式为[{"label": "", "coor": [x0, y0, x1, y1]}, {}, ..]
    return:
        img: 图片data
    """

    img = cv2.imread(img_file)
    for bbox in bbox_cfg:
        label = bbox["label"]
        coor = bbox["coor"]

        ## 画矩形框
        cv2.rectangle(img, (int(coor[0]), int(coor[1])),
                      (int(coor[2]), int(coor[3])), (0, 0, 255), thickness=2)

        ## 标注lable
        cv2.putText(img, label, (int(coor[0])-5, int(coor[1])-5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), thickness=2)
    if is_show:
        cv2.imshow('head', img)
        cv2.waitKey(0)
    if is_write:
        save_name = img_file[:-4] + "_bbox" + img_file[-4:]
        cv2.imwrite(save_name, img)
    return img

def img2base64(img):
    """
    numpy的int数据转换为base64格式。
    """
    retval, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer)
    img_base64 = img_base64.decode()
    return img_base64

def base642img(img_base64):
    """
    输入base64格式数据，转为numpy的int数据。
    """
    img = base64.b64decode(str(img_base64))
    img = np.fromstring(img, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img

if __name__ == '__main__':
    img_file = "C:/Users/yuanhui/Desktop/hear/test/#0773_org.jpg"
    bbox_cfg = [{"label": "meter", "coor": [508, 218, 1430, 1080]},
                {"label": "6", "coor": [908, 919, 947, 974]},
                {"label": "5", "coor": [988, 918, 1030, 979]}]
    draw_bboxs(img_file, bbox_cfg, is_write=True, is_show=False)
