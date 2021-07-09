# -*- coding: UTF-8 -*-

import os
import glob
import cv2  # conda install opencv || pip install opencv-python
import ffmpeg  # pip install ffmpeg-python
import shutil
from PIL import Image
import xml.etree.ElementTree as ET
# from xml import etree
from moviepy.editor import VideoFileClip, concatenate_videoclips

def crop_img_base_label(img_file, xml_file, save_dir, class_file):
    """
    根据标签裁剪图片。
    args:
        img_file: jpg图片路径
        xml_file: 标签路径
        save_dir: 裁剪后图片保存目录
        class_file: 需要裁剪的目标类型
    """
    img_name = os.path.basename(img_file)
    img_cv = cv2.imread(img_file) # 读取图片
    ## 路径中有中文字符是可使用该方法读取图片
    # img_cv = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)  
    f_class = open(class_file, 'r')
    classes = f_class.read().strip('\n').split('\n')
    f_class.close()

    count = 0
    # tree = etree.ElementTree()
    root = ET.parse(xml_file).getroot() #加载xml结构

    os.makedirs(save_dir, exist_ok=True)

    for obj in root.iter('object'):
        name = obj.find('name').text
        if name in classes:
            xmlbox = obj.find('bndbox')
            xmin = xmlbox.find('xmin').text
            ymin = xmlbox.find('ymin').text
            xmax = xmlbox.find('xmax').text
            ymax = xmlbox.find('ymax').text
            obj_img = img_cv[int(ymin):int(ymax), int(xmin):int(xmax)]
            save_file = os.path.join(save_dir, img_name[:-4] + "_" +str(count) + "_" + name + ".jpg")
            count += 1
            cv2.imwrite(save_file ,obj_img)  # 保存裁剪图片
            

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


if __name__ == '__main__':
    img_file = "../../data/meter/test1/2021_4_27_meter_122.jpg"
    xml_file = "../../data/meter/test1/num/2021_4_27_meter_122.xml"
    class_file = "../../data/meter/test1/classes.txt"
    save_dir = "../../data/meter/test1/"
    crop_img_base_label(img_file, xml_file, save_dir, class_file)
