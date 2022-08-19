"""
使用 utdnn/inspection:cuda11.4-conda-cuml-opencv-gtk 镜像运行
"""

import numpy as np
import cv2
from lib_sift_match import sift_create
import glob
import os
import time
import json
from lib_image_ops import base642img, img2base64, img_chinese

def create_npy(dir_):
    """
    创建sift特征列表和文件名列表
    args:
        dir: 待提取特征的文件夹路径
    """
    feats = {}
    for img_file in glob.glob(os.path.join(dir_, "*.jpg")):
        img_name = os.path.basename(img_file)
        img = cv2.imread(img_file)
        _, feat = sift_create(img)
        feats[img_name] = feat
    np.save("feats.npy", feats)

def sift_match_good(feat1, feat2, ratio=0.5, ops="Affine"):
    """
    使用sift特征，flann算法计算两张轻微变换的图片的的偏移转换矩阵M。
    args:
        feat1, feat2: 两图片的sift特征
        ratio: sift点正匹配的阈值
        ops: 变换的方式，可选择"Affine"(仿射), "Perspective"(投影)
    return:
        good_n: good match的数量
    """

    if  feat1 is None or feat2 is None or len(feat1) == 0 or len(feat2) == 0:
        print("warning: img have no sift feat!")
        return 0
    ## 画出siftt特征点
    # ref_sift = cv2.drawKeypoints(ref_img,kps1,ref_img,color=(255,0,255)) # 画sift点
    # tar_sift = cv2.drawKeypoints(tag_img,kps2,tag_img,color=(255,0,255))
    # hmerge = np.hstack((ref_sift, tar_sift)) # 两张图拼接在一起
    # cv2.imwrite("images/test_sift.jpg", hmerge)
    
    ## flann 快速最近邻搜索算法，计算两张特征的正确匹配点。
    ## https://www.cnblogs.com/shuimuqingyang/p/14789534.html
    ## 使用gpu计算sift matches
    feat1_gpu = cv2.cuda_GpuMat()
    feat1_gpu.upload(feat1) # 将数据转为cuda形式
    feat2_gpu = cv2.cuda_GpuMat()
    feat2_gpu.upload(feat2)
    matcherGPU = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_L2)
    matches = matcherGPU.knnMatch(feat1_gpu, feat2_gpu, k=2)

    # 画出匹配图
    # img_match = cv2.drawMatchesKnn(ref_img,kps1,tag_img,kps2,matches,None,flags=2)
    # cv2.imwrite("images/test_match.jpg",img_match)

    ## store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    # print("num of good match pointer:", len(good))
    return len(good)

if __name__ == '__main__':
    # create_npy("/data/PatrolAi/patrol_ai/python_codes/test/yuantu")
    feats = np.load("feats.npy", allow_pickle=True).item()
    in_dir = "/data/PatrolAi/patrol_ai/python_codes/test/paishe"

    

    







