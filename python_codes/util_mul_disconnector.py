import glob
import os
import cv2
import time
import numpy as np
import shutil
from app_inspection_disconnector import correct_offset, my_ssim, convert_coor


dir_ = "disconnector"
bboxes = [[261, 184, 307, 255],
          [260, 364, 307, 440],
          [326, 145, 364, 205],
          [327, 330, 362, 393]]

## 提取所有模板的sift特征
open_files = glob.glob(os.path.join(dir_,"open","*.jpg"))
close_files = glob.glob(os.path.join(dir_, "close", "*.jpg"))
sift = cv2.SIFT_create() # 创建sift对象
open_kpses = []
open_feats = []
for open_file in open_files:
    img = cv2.cvtColor(cv2.imread(open_file), cv2.COLOR_RGB2GRAY)
    kps, feat = sift.detectAndCompute(img, None)
    open_kpses.append(kps); open_feats.append(feat)
close_kpses = []
close_feats = []
for close_file in close_files:
    img = cv2.cvtColor(cv2.imread(close_file), cv2.COLOR_RGB2GRAY)
    kps, feat = sift.detectAndCompute(img, None)
    close_kpses.append(kps); close_feats.append(feat)

## flann 快速最近邻搜索算法，计算两张特征的正确匹配点。
## https://www.cnblogs.com/shuimuqingyang/p/14789534.html
flann_index_katree = 1
index_params = dict(algorithm=flann_index_katree, trees=5) # trees:指定待处理核密度树的数量
search_params = dict(checks=50) # checks: 指定递归遍历迭代的次数
flann = cv2.FlannBasedMatcher(index_params, search_params)


while True:
    ## 调用摄像头采集图片
    cap = cv2.VideoCapture('rtsp://admin:ut0000@192.168.57.147/mpeg4/ch1/sub/av_stream') # 读取网络相机 'rtsp://admin:ut0000@192.168.57.25'
    _, img_tag = cap.read() #读取最新帧
    time_id = time.strftime("%m%d%H%M%S")
    cap.release()

    ## 创建保存结果文件夹
    save_dir = os.path.join(dir_, "result", time_id)
    os.makedirs(save_dir, exist_ok=True)

    ## 提取图片sift特征、
    img = cv2.cvtColor(img_tag, cv2.COLOR_RGB2GRAY)
    kps_tag, feat_tag = sift.detectAndCompute(img, None)

    ## 特征点匹配
    ratio = 0.5
    open_id = 0
    open_good = []
    for i, feat_ref in enumerate(open_feats):
        matches = flann.knnMatch(feat_ref, feat_tag, k=2)
        good = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good.append(m)
        if len(good) > len(open_good):
            open_id = i
            open_good = good

    close_id = 0
    close_good = []
    for i, feat_open in enumerate(close_feats):
        matches = flann.knnMatch(feat_ref, feat_tag, k=2)
        good = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good.append(m)
        if len(good) > len(close_good):
            close_id = i
            close_good = good

    ## 根据匹配的id确定模板图
    open_file = open_files[open_id]
    kps_open = open_kpses[open_id]
    feat_open = open_feats[open_id]
    close_file = close_files[close_id]
    kps_close = close_kpses[close_id]
    feat_close = close_feats[close_id]

    ## 求偏移矩阵
    min_match_count = 10
    if len(open_good) > min_match_count:
        src_pts = np.float32([kps_open[m.queryIdx].pt for m in open_good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps_tag[m.trainIdx].pt for m in open_good]).reshape(-1, 1, 2)
        M_open, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5)
    else:
        print("Open_good not enough matches are found - {}/{}".format(len(open_good), min_match_count))
        M_open = None
    # if len(close_good) > min_match_count:
    #     src_pts = np.float32([kps_open[m.queryIdx].pt for m in close_good]).reshape(-1, 1, 2)
    #     dst_pts = np.float32([kps_tag[m.trainIdx].pt for m in close_good]).reshape(-1, 1, 2)
    #     M_close, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5)
    # else:
    #     print("Open_good not enough matches are found - {}/{}".format(len(open_good), min_match_count))
    #     M_close = None

    ## 根据偏移矩阵矫正图片
    img_open = cv2.imread(open_file)
    img_close = cv2.imread(close_file)
    img_tag_warped = correct_offset(img_open, img_tag, M_open)

    out_open = os.path.join(save_dir, "open_"+os.path.basename(open_file))
    out_close = os.path.join(save_dir, "close_"+os.path.basename(close_file))
    out_tag = os.path.join(save_dir, "tag_"+time_id+".jpg")
    shutil.copy(open_file, out_open)
    shutil.copy(close_file, out_close)
    cv2.imwrite(out_tag,img_tag)

    f = open(os.path.join(save_dir, "result_"+time_id+".txt"),"w",encoding='utf-8')
    for bbox in bboxes:
        img_op = img_open[bbox[1]: bbox[3], bbox[0]: bbox[2]]
        img_cl = img_close[bbox[1]: bbox[3], bbox[0]: bbox[2]]
        img_tag_warp = img_tag_warped[bbox[1]: bbox[3], bbox[0]: bbox[2]]
        score_open = my_ssim(img_tag_warp, img_op) #计算ssim结构相似性
        score_close = my_ssim(img_tag_warp, img_cl)
        if score_close < score_open:
            result = 1
            label = "Open( %.3f : %.3f )" % (score_open, score_close)
            color = (0, 255, 0)
        else:
            result = 0
            color = (0, 0, 255)
            label = "Close( %.3f : %.3f )" % (score_open, score_close)
        
        f.write(label+"\n")
        cv2.rectangle(img_open, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 225, 0), thickness=1)
        cv2.rectangle(img_close, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), thickness=1)
        lines = [[(bbox[0], bbox[1]), (bbox[2], bbox[1])],
                [(bbox[2], bbox[1]), (bbox[2], bbox[3])],
                [(bbox[2], bbox[3]), (bbox[0], bbox[3])],
                [(bbox[0], bbox[1]), (bbox[0], bbox[3])]]
        for line in lines:
            line = [convert_coor(line[0], M_open), convert_coor(line[1], M_open)]
            cv2.line(img_tag, line[0], line[1], color, 1)
        point = convert_coor(lines[0][0], M_open)
        cv2.putText(img_tag, label, (point[0]-10, point[1]-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)
    
    f.close()
    cv2.imwrite(out_open[:-4]+"_cfg.jpg", img_open)
    cv2.imwrite(out_close[:-4]+"_cfg.jpg", img_close)
    cv2.imwrite(out_tag[:-4]+"_cfg.jpg", img_tag)

    ## 10分钟一次
    time.sleep(600)






