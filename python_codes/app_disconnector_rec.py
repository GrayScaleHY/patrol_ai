from sys import int_info
import cv2
import numpy as np
import os
import math
from skimage import measure # pip install scikit-image
import base64
from lib_image_ops import base642img
import time


def my_ssim(img1, img2):
    """
    python官方的计算ssim的包。
    """

    ## 使用tensorflow计算ssim, 输入彩图
    # score = tf.image.ssim(img1, img2, 255)
    # score = score.numpy()

    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    score = measure.compare_ssim(img1, img2) #输入灰度图
    
    return score


def sift_match(ref_img, tag_img, ratio=0.5, ops="Affine"):
    """
    使用sift特征，flann算法计算两张轻微变换的图片的的偏移转换矩阵M。
    args:
        ref_img: 参考图片data
        tag_img: 变换图片data
        ratio: sift点正匹配的阈值
        ops: 变换的方式，可选择"Affine"(仿射), "Perspective"(投影)
    return:
        M:偏移矩阵, 2*3矩阵，前两列表示仿射变换，后一列表示平移量。
          偏移后的点的计算公式：(x', y') = M * (x, y, 1)
    """

    ## 彩图转灰度图，灰度图是二维的
    if len(ref_img.shape) == 3:
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
    if len(tag_img.shape) == 3:
        tag_img = cv2.cvtColor(tag_img, cv2.COLOR_RGB2GRAY)

    ## 直方图均衡化，增加图片的对比度
    # ref_img = cv2.equalizeHist(ref_img)
    # tag_img = cv2.equalizeHist(tag_img)

    print("ref_img shape:", ref_img.shape)
    print("tag_img shape:", tag_img.shape)

    ## 提取sift特征 
    sift = cv2.SIFT_create() # 创建sift对象
    # kps: 关键点，包括 angle, class_id, octave, pt, response, size
    # feat: 特征值，每个特征点的特征值是128维
    kps1, feat1 = sift.detectAndCompute(ref_img, None) #提取sift特征
    kps2, feat2 = sift.detectAndCompute(tag_img, None)
    print("sift len of ref_img:", len(kps1))
    print("sift len of tag_img:", len(kps2))

    ## 画出siftt特征点
    # ref_sift = cv2.drawKeypoints(ref_img,kps1,ref_img,color=(255,0,255)) # 画sift点
    # tar_sift = cv2.drawKeypoints(tag_img,kps2,tag_img,color=(255,0,255))
    # hmerge = np.hstack((ref_sift, tar_sift)) # 两张图拼接在一起
    # cv2.imwrite("images/test_sift.jpg", hmerge)
    
    ## flann 快速最近邻搜索算法，计算两张特征的正确匹配点。
    ## https://www.cnblogs.com/shuimuqingyang/p/14789534.html
    flann_index_katree = 1
    index_params = dict(algorithm=flann_index_katree, trees=5) # trees:指定待处理核密度树的数量
    search_params = dict(checks=50) # checks: 指定递归遍历迭代的次数
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(feat1, feat2, k=2)
    # 画出匹配图
    # img_match = cv2.drawMatchesKnn(ref_img,kps1,tag_img,kps2,matches,None,flags=2)
    # cv2.imwrite("images/test_match.jpg",img_match)

    ## store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    print("num of good match pointer:", len(good))

    ## 求偏移矩阵,[[^x1, ^x2, dx],[^y1, ^y2, dy]]
    min_match_count = 10  ## 至少多少个好的匹配点
    if len(good) > min_match_count:
        src_pts = np.float32([kps1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        ## 根据正匹配点求偏移矩阵
        if ops == "Affine":
            M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5)
        elif ops == "Perspective":
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, maxIters=2000, confidence=0.995)
        else:
            print("ops is wrong!")
            M = cv2.estimateRigidTransform(src_pts, dst_pts, False)  # python bindings removed

        print("Enough matches are found - {}/{}".format(len(good), min_match_count))
        return M
    else:
        print("Not enough matches are found - {}/{}".format(len(good), min_match_count))
        return None


def correct_offset(ref_img, tag_img, bbox=None, ops="Affine"):
    """
    根据偏移矩阵矫正图片。
    args:
        ops: 变换的方式，可选择"Affine"(仿射), "Perspective"(投影)
    return:
        img_tag_warped: 矫正之后的tag_img
        coors_tar: ref_img上的bbox相对与tag_img的bbox。
    """
    ## 使用sift特征，flann算法计算两张轻微变换的图片的的偏移转换矩阵M。
    M = sift_match(ref_img, tag_img, ops=ops).astype(np.float32) # 2*3偏移矩阵

    if M is None:
        return None, None
    
    ## 矫正图片
    if M.shape == (2, 3):  # warp affine
        img_tag_warped = cv2.warpAffine(tag_img, M, (tag_img.shape[1], tag_img.shape[0]), flags=cv2.WARP_INVERSE_MAP)
    elif M.shape == (3, 3):  # warp perspective
        img_tag_warped = cv2.warpPerspective(tag_img, M, (tag_img.shape[1], tag_img.shape[0]), flags=cv2.WARP_INVERSE_MAP)

    if bbox is None:
        return img_tag_warped, None

    ## bbox变换， (x', y') = M * (x, y, 1)
    coors_ref = np.array([[bbox[0],bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]])
    coors_tar = np.c_[coors_ref, np.array([1]*4)] # 添加一列1象量
    coors_tar = np.transpose(coors_tar)
    coors_tar = np.dot(M, coors_tar) # 矩阵相乘
    coors_tar = np.transpose(coors_tar).astype(int)

    return img_tag_warped, coors_tar


def disconnector_rec(data):
    """
    刀闸识别
    """
    TIME_START = time.strftime("%m-%d-%H-%M-%S")
    save_dir = os.path.join(os.path.join("disconnector_result",TIME_START)) #保存图片的路径
    os.makedirs(save_dir, exist_ok=True)
    
    ## 提取data信息
    out_data = {"code": 0, "data":{}, "msg": "Success request pointer"}
    img_tag = base642img(data["image"])
    img_open = base642img(data["config"]["img_open"])
    img_close = base642img(data["config"]["img_close"])
    bbox = data["config"]["bbox"]
    cv2.imwrite(os.path.join(save_dir,"img_close.jpg"), img_close)
    cv2.imwrite(os.path.join(save_dir,"img_tag.jpg"), img_tag)
    cv2.imwrite(os.path.join(save_dir,"img_open.jpg"), img_open)

    ## 对图像做矫正偏移
    img_tag_warped, coors_tar = correct_offset(img_open, img_tag)
    if img_tag_warped is None:
        out_data["msg"] = "img_tar is not match img_open!"
        return out_data

    ## 截取图片区域，并且用ssim算法比较相似性
    img_op = img_open[bbox[1]: bbox[3], bbox[0]: bbox[2]]
    img_cl = img_close[bbox[1]: bbox[3], bbox[0]: bbox[2]]
    img_tag_warp = img_tag_warped[bbox[1]: bbox[3], bbox[0]: bbox[2]]
    score_open = my_ssim(img_tag_warp, img_op) #计算ssim结构相似性
    score_close = my_ssim(img_tag_warp, img_cl)

    ## 将计算结果复制到return data中。
    if score_close > score_open:
        result = 1
    else:
        result = 0
    out_data["data"] = {"result": result, "score_open": score_open, "score_close": score_close}

    label_s = "op : cl = %.3f : %.3f" % (score_open, score_close)
    print(label_s)

    ## 画图，将结果用图片的形式展示结果，有利于debug
    cv2.imwrite(os.path.join(save_dir,"img_op.jpg"), img_op)
    cv2.imwrite(os.path.join(save_dir,"img_cl.jpg"), img_cl)
    cv2.imwrite(os.path.join(save_dir,"img_tag_warp.jpg"), img_tag_warp)
    cv2.rectangle(img_open, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), thickness=2)
    cv2.rectangle(img_tag_warped, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), thickness=2)
    cv2.putText(img_tag_warped, label_s, (bbox[0]-5, bbox[1]-5),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), thickness=2)
    cv2.imwrite(os.path.join(save_dir,"img_open.jpg"), img_open)
    cv2.imwrite(os.path.join(save_dir,"img_tag_warped.jpg"), img_tag_warped)

    return out_data


def init_data():
    ref_file = "C:/data/disconnector/test_0_open_open.png"
    close_file = "C:/data/disconnector/test_0_close_open.png"
    open_file = "C:/data/disconnector/test_0_open_close.png"
    bbox =  [1460, 405, 1573, 578]
    with open(ref_file, "rb") as imageFile:
        img_tag= imageFile.read()
    img_tag = base64.b64encode(img_tag).decode('utf-8')
    with open(close_file, "rb") as imageFile:
        img_close = imageFile.read()
    img_close = base64.b64encode(img_close).decode('utf-8')
    with open(open_file, "rb") as imageFile:
        img_open = imageFile.read()
    img_open = base64.b64encode(img_open).decode('utf-8')

    data = {
        "image": img_tag,
        "config": {"img_open": img_open, "img_close": img_close, "bbox": bbox},
        "type": "disconnector"
    }
    return data

if __name__ =='__main__':
    
    data = init_data()
    out_data = disconnector_rec(data)
    




