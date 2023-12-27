# -*- coding: UTF-8 -*-

import os
import glob
from sys import exc_info
import cv2  # conda install opencv || pip install opencv-python
import ffmpeg  # pip install ffmpeg-python
import shutil
from PIL import Image, ExifTags, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, concatenate_videoclips
import random
import base64
import numpy as np
import json
try:
    from imagededup.methods import PHash # pip install imagededup
    from imagededup.methods import CNN
except:
    print("Warning: imagededup package is not exist!")

def duplicates_phash(img_dir):
    '''
    使用Phash寻找文件夹中的重复图片
    https://github.com/idealo/imagededup
    args:
        img_dir: 存放图片的文件夹路径
    return:
        duplicates: 描述重复关系的字典。格式：{"name1": ["name2", "name3"], "name2": ["name1", "name3"]}
    '''
    print("de duplicates with PHash")
    phasher = PHash()
    encodings = phasher.encode_images(image_dir=img_dir)
    duplicates = phasher.find_duplicates(encoding_map=encodings)
    return duplicates

def duplicates_cnn(img_dir, min_similarity_threshold=0.97):
    '''
    使用CNN寻找文件夹中的重复图片
    https://github.com/idealo/imagededup/blob/master/examples/Evaluation.ipynb
    args:
        img_dir: 存放图片的文件夹路径
    return:
        duplicates: 描述重复关系的字典。格式：{"name1": [("name2", score), ("name3", score)]}
    '''
    print("de duplicates with CNN")
    cnn_encoder = CNN()
    duplicates = cnn_encoder.find_duplicates(image_dir=img_dir, scores=True, min_similarity_threshold=min_similarity_threshold)
    return duplicates

def process_duplicates_images(img_dir, method="Phash", conf=0.97):
    """
    处理需要去重的文件夹
    args:
        img_dir: 存放图片的文件夹路径
        matod: 选择去重的算法，可选"CNN"或"Phash"
    """
    # 创建diff和same文件夹
    diff_dir = os.path.join(img_dir, "diff")
    same_dir = os.path.join(img_dir, "same")
    os.makedirs(diff_dir, exist_ok=True)
    os.makedirs(same_dir, exist_ok=True)

    # 求出文件夹中重复的图片
    if method == "CNN":
        d_ = duplicates_cnn(img_dir, min_similarity_threshold=conf)
        duplicates = {}
        for name in d_:
            duplicates[name] = [(i[0], float(round(i[1], 3))) for i in d_[name]]
    else:
        d_ = duplicates_phash(img_dir)
        duplicates = {}
        for name in d_:
            duplicates[name] = [(i, 1.0) for i in d_[name]]

    
    # 打包字典
    out_dict = {}
    for name in duplicates:
        if len(list(set([i for i in out_dict]) & set([i[0] for i in duplicates[name]]))) == 0:
            out_dict[name] = duplicates[name]

    ## 将重复的文件名对应关系保存成字典
    f = open(os.path.join(img_dir, "duplicates.json"), "w", encoding='utf-8')
    json.dump(out_dict, f, ensure_ascii=False, indent=2)
    f.close()

    # 移动图片到diff或same文件夹
    for diff_name in out_dict:
        diff_file = os.path.join(img_dir, diff_name)
        diff_out = os.path.join(diff_dir, diff_name)
        os.rename(diff_file, diff_out)

        for same_ in out_dict[diff_name]:
            same_file = os.path.join(img_dir, same_[0])
            if os.path.exists(same_file):
                score = same_[1]
                if score > conf:
                    same_out = os.path.join(same_dir, same_[0])
                else:
                    same_out = os.path.join(diff_dir, same_[0])
                os.rename(same_file, same_out)

def get_exif_info(img_file, tag='Orientation'):
    """
    获取图片的exif信息。
    """
    item = None
    ## 用循环查找自己需要的信息的item。
    for it in ExifTags.TAGS.keys():
        if ExifTags.TAGS[it]==tag:
            item = it
            break 
    if item is None:
        return None

    img = Image.open(img_file)
    return img._getexif()[item]

def img_rotate_batch(dir):
    """
    将文件夹中带旋转信息的图片进行旋转。防止标注错误。
    """
    count_all = 0
    count_rotate = 0
    for root, dirs, files in os.walk(dir):
        
        for file_name in files:
            if not file_name.endswith((".jpg",".JPG",".png",".PNG",".bmp")):
                continue
            count_all += 1

            img_file = os.path.join(root, file_name)

            img = Image.open(img_file)
            try:
                exif = img._getexif()
            except:
                print("Warning:",img_file," AttributeError: BmpImageFile object has no attribute _getexif")
                continue
            if exif == None or 274 not in exif:
                img.close()
                continue
            if exif[274] == 3: ## 274是旋转信息的id
                angle = 180
            elif exif[274] == 6:
                angle = 270
            elif exif[274] == 8:
                angle = 90
            else:
                angle = 0
            if angle != 0:
                img=img.rotate(angle, expand=True)
                print(img_file,"rotated", angle, "already!")
                count_rotate += 1
                img.save(img_file)
            img.close()
    print("num of all img:", count_all)
    print("num of rotate img:", count_rotate)

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

    print("format_name:{} /ncodec_name:{} /nduration_ts:{} /nwidth:{} /nheight:{} /nfps:{}".format(
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


def img_chinese(img, text, coor, color=(0, 255, 0), size=20):
    """
    给cv2读取的图片标上中文。
    args:
        img: cv2读取的图片data
        text: 中文内容
        coor: 起始坐标
    return:
        img
    """
    color_real = (color[2], color[1], color[0])
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    # if size > 50:
    #     size = 50
    fontStyle = ImageFont.truetype(
        "../c_codes/simsun.ttc", size, index=1, encoding="utf-8")
    # 绘制文本
    draw.text(coor, text, color_real, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def img_equalize(img):
    """
    直方图均衡化
    """
    if len(img.shape) == 3:
        for i in range(3):
            img[:,:,i] = cv2.equalizeHist(img[:,:,i])
    elif len(img.shape) == 2:
        img = cv2.equalizeHist(img)
    else:
        print("Warning: img is not a matrix !")
    return img


def img_clahe(img):
    """
    自适应均衡化
    """
    clahe = cv2.createCLAHE(clipLimit=2.0,
                        tileGridSize=(8, 8))
    if len(img.shape) == 3:
        for i in range(3):
            clahe.apply(img[:,:,i])
    elif len(img.shape) == 2:
        clahe.apply(img)
    return img

if __name__ == '__main__':
    # img_file = "C:/Users/yuanhui/Desktop/hear/test/#0773_org.jpg"
    # bbox_cfg = [{"label": "meter", "coor": [508, 218, 1430, 1080]},
    #             {"label": "6", "coor": [908, 919, 947, 974]},
    #             {"label": "5", "coor": [988, 918, 1030, 979]}]
    # draw_bboxs(img_file, bbox_cfg, is_write=True, is_show=False)
    # os.path.exists

    img_rotate_batch("/data/PatrolAiImg2022/bjbmyw")
