import cv2
import numpy as np
import os
import math

def sift_match(ref_img, tar_img, ratio=0.5):
    """
    使用sift特征，flann算法计算两张轻微变换的图片的的偏移转换矩阵M。
    args:
        ref_img: 参考图片data
        tar_img: 变换图片data
        ratio: sift点正匹配的阈值
    return:
        M:偏移矩阵, 2*3矩阵，前两列表示仿射变换，后一列表示平移量。
          偏移后的点的计算公式：(x', y') = M * (x, y, 1)
    """
    
    ## 彩图转灰度图，灰度图是二维的
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
    tar_gray = cv2.cvtColor(tar_img, cv2.COLOR_RGB2GRAY)

    ## 直方图均衡化，增加图片的对比度
    # ref_gray = cv2.equalizeHist(ref_gray)
    # tar_gray = cv2.equalizeHist(tar_gray)

    print("ref_gray shape:", ref_gray.shape)
    print("tar_gray shape:", tar_gray.shape)

    ## 提取sift特征 
    sift = cv2.SIFT_create() # 创建sift对象
    # kps: 关键点，包括 angle, class_id, octave, pt, response, size
    # feat: 特征值，每个特征点的特征值是128维
    kps1, feat1 = sift.detectAndCompute(ref_gray, None) #提取sift特征
    kps2, feat2 = sift.detectAndCompute(tar_gray, None)
    print("sift len of ref_gray:", len(kps1))
    print("sift len of tar_gray:", len(kps2))

    ## 画出siftt特征点
    # ref_sift = cv2.drawKeypoints(ref_img,kps1,ref_img,color=(255,0,255)) # 画sift点
    # tar_sift = cv2.drawKeypoints(tar_img,kps2,tar_img,color=(255,0,255))
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
    # img_match = cv2.drawMatchesKnn(ref_img,kps1,tar_img,kps2,matches,None,flags=2)
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
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5)
        # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, maxIters=2000, confidence=0.995)
        # M = cv2.estimateRigidTransform(src_pts, dst_pts, False)  # python bindings removed
        print("Enough matches are found - {}/{}".format(len(good), min_match_count))
        return M
    else:
        print("Not enough matches are found - {}/{}".format(len(good), min_match_count))
        return None

def correct_offset(ref_img, tar_img, bbox=None):
    """
    根据偏移矩阵矫正图片。
    return:
        warped_img: 矫正之后的tar_img
        coors_tar: ref_img上的bbox相对与tar_img的bbox。
    """
    ## 使用sift特征，flann算法计算两张轻微变换的图片的的偏移转换矩阵M。
    M = sift_match(ref_img, tar_img).astype(np.float32) # 2*3偏移矩阵

    if M == None:
        return None, None
    
    ## 矫正图片
    if M.shape == (2, 3):  # warp affine
        warped_img = cv2.warpAffine(tar_img, M, (tar_img.shape[1], tar_img.shape[0]), flags=cv2.WARP_INVERSE_MAP)
    elif M.shape == (3, 3):  # warp perspective
        warped_img = cv2.warpPerspective(tar_img, M, (tar_img.shape[1], tar_img.shape[0]), flags=cv2.WARP_INVERSE_MAP)

    if bbox==None:
        return warped_img, None

    ## bbox变换， (x', y') = M * (x, y, 1)
    coors_ref = np.array([[bbox[0],bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]])
    coors_tar = np.c_[coors_ref, np.array([1]*4)] # 添加一列1象量
    coors_tar = np.transpose(coors_tar)
    coors_tar = np.dot(M, coors_tar) # 矩阵相乘
    coors_tar = np.transpose(coors_tar).astype(int)

    return warped_img, coors_tar
    
if __name__ =='__main__':
    coor = [960, 346, 1100, 748]
    ref_img = cv2.imread("images/test_0.jpg")
    tar_img = cv2.imread("images/test_2.jpg")
    warped_img, coors_tar = correct_offset(ref_img, tar_img, bbox=coor)
    cv2.imwrite("images/test_warp.jpg", warped_img)

    



