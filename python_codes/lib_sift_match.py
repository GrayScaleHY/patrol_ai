"""
sift匹配相关的算法函数
"""

import cv2
import numpy as np
import time
try:
    from skimage.measure import compare_ssim as sk_cpt_ssim # pip install scikit-image
except:
    from skimage.metrics import structural_similarity as sk_cpt_ssim
from lib_image_ops import img_chinese


def _resize(img):
    """
    return:
        img: image data
        rate: 伸缩率
    """
    H, W = img.shape[:2]
    if H > W:
        rate = H / 640
    else:
        rate = W / 640
    img = cv2.resize(img, (int(W / rate), int(H / rate)))
    return img, rate


def _resize_feat(img):
    """
    return: 
        img_resize: resize之后的图片
        rate: resize之后的伸缩率
        feat: sift特征
    """
    img_resize, rate = _resize(img)
    feat = sift_create(img_resize)
    return img_resize, rate, feat


def my_ssim(img1, img2):
    """
    python官方的计算ssim的包。
    ## 
    """
    # 使用tensorflow计算ssim, 输入彩图,需要较长时间。
    # score = tf.image.ssim(img1, img2, 255)
    # 注意需要根据图片长宽来设置filter_size的大小，如下链接。
    ## https://github.com/tensorflow/tensorflow/issues/33840#issuecomment-633715778
    # score = tf.image.ssim_multiscale(img1, img2, 255, power_factors=(0.0448, 0.2856, 0.3001),filter_size=6)
    # score = score.numpy()

    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    score = sk_cpt_ssim(img1, img2) #输入灰度图 , multichannel=True
    return score


def sift_create(img):
    """
    提取sift特征
    return: (kps, feat)
    """
    ## 彩图转灰度图，灰度图是二维的
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    ## 直方图均衡化，增加图片的对比度
    # img = cv2.equalizeHist(img)

    ## 提取sift特征 
    sift = cv2.SIFT_create() # 创建sift对象

    # sift = cv2.xfeatures2d.SIFT_create()
    # kps: 关键点，包括 angle, class_id, octave, pt, response, size
    # feat: 特征值，每个特征点的特征值是128维
    start = time.time()
    kps, feat = sift.detectAndCompute(img, None) #提取sift特征
    print("sift len of img:", len(kps))
    return (kps, feat)


def sift_match(feat_ref, feat_tag, rm_regs=[], ratio=0.5, ops="Affine"):
    """
    使用sift特征，flann算法计算两张轻微变换的图片的的偏移转换矩阵M。
    args:
        feat_ref: 参考图片的sift特征，格式为：(kps, feat)
        feat_tag: 待分析图片的sift特征，格式为：(kps, feat)
        rm_regs: 需要去掉sift特征的区域，例如OSD区域。格式为[[xmin, xmax, ymin, ymax], ..]
        ratio: sift点正匹配的阈值
        ops: 变换的方式，可选择"Affine"(仿射), "Perspective"(投影)
    return:
        M:偏移矩阵, 2*3矩阵，前两列表示仿射变换，后一列表示平移量。
          偏移后的点的计算公式：(x', y') = M * (x, y, 1)
    """
    kps1, feat1 = feat_ref
    kps2, feat2 = feat_tag

    ## 将rm_regs区域中的sift特征点去除
    if len(rm_regs) > 0:
        rm_ids = []
        for reg in rm_regs:
            for i in range(len(kps1)):
                pt_ = kps1[i].pt
                if reg[0] < pt_[0] < reg[2] and reg[1] < pt_[1] < reg[3]:
                    rm_ids.append(i)
        kps = []
        for i in range(len(kps1)):
            if i not in rm_ids:
                kps.append(kps1[i])
        kps1 = kps
        feat1 = np.delete(feat1, rm_ids, axis=0)
        rm_ids = []
        for reg in rm_regs:
            for i in range(len(kps2)):
                pt_ = kps2[i].pt
                if reg[0] < pt_[0] < reg[2] and reg[1] < pt_[1] < reg[3]:
                    rm_ids.append(i)
        kps = []
        for i in range(len(kps2)):
            if i not in rm_ids:
                kps.append(kps2[i])
        kps2 = kps
        feat2 = np.delete(feat2, rm_ids, axis=0)

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


def correct_offset(tag_img, M):
    """
    根据偏移矩阵矫正图片。
    args:
        M: 偏移矩阵
    return:
        img_tag_warped: 矫正之后的tag_img
    """
    if M is None:
        return tag_img
    
    ## 矫正图片
    if M.shape == (2, 3):  # warp affine
        img_tag_warped = cv2.warpAffine(tag_img, M, (tag_img.shape[1], tag_img.shape[0]), flags=cv2.WARP_INVERSE_MAP)
    elif M.shape == (3, 3):  # warp perspective
        img_tag_warped = cv2.warpPerspective(tag_img, M, (tag_img.shape[1], tag_img.shape[0]), flags=cv2.WARP_INVERSE_MAP)

    return img_tag_warped


def convert_coor(coor_ref, M):
    """
    使用偏移矩阵M计算参考坐标发生偏移后的对应坐标。
    args:
        coor_ref: 参考坐标
        M: 偏移矩阵
    return: 
        (x, y): 转换后的坐标
    """
    if M is None:
        return coor_ref

    M = np.array(M, dtype=float)
    
    assert M.shape == (2, 3) or M.shape == (3, 3), "shape of M is not match !"

    coor_ref = np.array(list(coor_ref) + [1], dtype=float)
    coor_tag = np.dot(M, coor_ref) # (2, 3)的转换矩阵直接通过相乘得到转换后坐标

    if M.shape == (3, 3): # Homo坐标系
        x = coor_tag[0] / coor_tag[2]; y = coor_tag[1] / coor_tag[2]
        coor_tag = np.array([x, y], dtype=float)

    return tuple(coor_tag.astype(int))


def detect_diff(img_ref, feat_ref, img_tag, feat_tag):
    """
    判别算法，检测出待分析图与基准图的差异区域
    return:
        rec_real: 差异区域，若判定没差异，则返回[]
    """
    img_tag_ = img_tag.copy()
    H,W = img_tag.shape[:2]
    ## 求偏移矩阵
    M = sift_match(feat_tag, feat_ref, ratio=0.5, ops="Affine")

    ## 对待分析图进行纠偏
    ref_warped = correct_offset(img_ref, M)
    # cv2.imwrite("test1/ref_swarped.jpg", ref_warped)

    ## 将矫正图与外边缘切掉,并且将基准图的相应位置切掉
    coors = [[0,0],[W,0],[0,H],[W,H]]
    off_c = []
    for c in coors:
        try:
            d = np.linalg.inv(M[:,:-1])
        except:
            return []
        e = np.array([c[0]-M[0][-1], c[1] - M[1][-1]])
        f = np.dot(d, e)
        off_c.append([f[0], f[1]])
    off_c = np.array(off_c,dtype=int)
    xmin = max(0, off_c[0][0], off_c[2][0])
    ymin = max(0, off_c[0][1], off_c[1][1])
    xmax = min(W, off_c[1][0], off_c[3][0])
    ymax = min(H, off_c[2][1], off_c[3][1])
    rec_cut = [xmin, ymin, xmax, ymax]
    img_ref = ref_warped[ymin:ymax, xmin:xmax, :]
    img_tag = img_tag[ymin:ymax, xmin:xmax, :]
    # cv2.imwrite("test1/tag_cut.jpg", img_tag)
    # cv2.imwrite("test1/ref_cut.jpg", img_ref)

    ## 将图片转为灰度图后相减，再二值化，得到差异图片
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY)
    if len(img_tag.shape) == 3:
        img_tag = cv2.cvtColor(img_tag, cv2.COLOR_RGB2GRAY)
    dif_img = img_tag.astype(float) - img_ref.astype(float)
    dif_img = np.abs(dif_img).astype(np.uint8)
    _, dif_img = cv2.threshold(dif_img, 100, 255, cv2.THRESH_BINARY) # 二值化
    # cv2.imwrite("test1/tag_diff.jpg",dif_img)

    ## 对差异性图片进行腐蚀操作，去除零星的点。
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,5))
    dif_img = cv2.erode(dif_img,kernel,iterations=1)
    # cv2.imwrite("test1/tag_diff_erode.jpg",dif_img)

    ## 用最小外接矩阵框出差异的地方
    H, W = dif_img.shape
    contours, _ = cv2.findContours(dif_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # 根据连通域求每个轮廓
    list_ = []
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont) # 轮廓的最小外接矩阵
        list_.append([x, y, x+w, y+w])
    if len(list_) > 0:
        list_ = np.array(list_)
        xmin = np.min(list_[:,0])
        ymin = np.min(list_[:,1])
        xmax = np.max(list_[:,2])
        ymax = np.max(list_[:,3])
        rec_dif = [xmin, ymin, xmax, ymax]
    else:
        return []

    # cv2.rectangle(dif_img, (rec_dif[0], rec_dif[1]), (rec_dif[2], rec_dif[3]), (255), 1)
    # cv2.imwrite("test1/tag_diff_rec.jpg",dif_img)

    ## 将矩形框映射回原待分析图
    rec_real = [rec_dif[0] + rec_cut[0], rec_dif[1] + rec_cut[1], 
                rec_dif[2]+ rec_cut[0], rec_dif[3] + rec_cut[1]]

    # cv2.rectangle(img_tag_, (rec_real[0], rec_real[1]), (rec_real[2], rec_real[3]), (0,0,255), 5)
    # cv2.imwrite("test1/tag_rec_real.jpg", img_tag_)

    return rec_real

if __name__ == '__main__':
    ref_file = "test/test1/0001_normal_off.jpg"
    tag_file = "test/test1/0001_normal_on.jpg"
    img_ref = cv2.imread(ref_file)
    img_tag = cv2.imread(tag_file)

    feat_ref = sift_create(img_ref)
    feat_tag = sift_create(img_tag)
    
    rec_real = detect_diff(img_ref, feat_ref, img_tag, feat_tag)