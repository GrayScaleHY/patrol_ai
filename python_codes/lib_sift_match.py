"""
sift匹配相关的算法函数
"""

import cv2
import numpy as np
import time
try:
    from skimage.measure import compare_ssim as sk_cpt_ssim # pip install scikit-image
except:
    from skimage.metrics import structural_similarity as sk_cpt_ssim # pip install scikit-image
from lib_image_ops import img_chinese
import pyrtools as pt  # pip install pyrtools
from scipy import signal
from scipy.ndimage import uniform_filter, gaussian_filter
from collections import Counter
from skimage import exposure
import cupy as cp ## pip install cupy-cuda114
import math

try:
    ## https://rapids.ai/start.html#get-rapids
    ## https://docs.rapids.ai/api/cuml/stable/api.html?highlight=kmeans#cuml.KMeans
    import cudf
    from cuml import PCA
    from cuml.cluster import KMeans
    kmeans_lib = "cuml"
    print("Notice: PCA and KMeans use cuml !!!")
except:
    try:
        ## https://github.com/h2oai/h2o4gpu
        from h2o4gpu.solvers.pca import PCAH2O as PCA
        from h2o4gpu import KMeans
        kmeans_lib = "h2o4gpu"
        print("Notice: PCA and KMeans use h2o4gpu !!!")
    except:
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        kmeans_lib = "sklearn"
        print("Notice: PCA and KMeans use sklearn !!!")

try:
    ## https://github.com/zldrobit/pycudasift/tree/Maxwell-fix
    ## https://github.com/zldrobit/pycudasift/issues/1
    import cudasift
    sift_lib = "cudasift"
    print("Notice: sift create use cudasift !!!")
except:
    sift_lib = "opencv"
    print("Notice: sift create use cv2 !!!")

try:
    from pack_vector_set import pack_vector_set
    vector_lib = "cython"
    print("Notice: vector_set use cython !!!")
except:
    vector_lib = "python"
    print("Notice: vector_set use python !!!")

if sift_lib == "cudasift":
    sift_data = cudasift.PySiftData(25000)

if kmeans_lib == "cuml":
    kmeans = KMeans(n_clusters=2, max_iter=100, tol=1e-2)
else:
    kmeans = KMeans(2, verbose = 0, max_iter=100, tol=1e-2)

cuda_source = open('change_map.cu').read()
module=cp.RawModule(code=cuda_source)
cuda_change_map = module.get_function('cuda_change_map')

def get_change_map(im1, im2, tol=0.02, return_numpy=True):
    BLOCK_SIZE = 256
    assert im1.shape == im2.shape

    height, width = im1.shape
    xm = int(width * tol)
    ym = int(height * tol)
    print(f"xm = {xm}, ym = {ym}, {height - 2 * ym}, {width - 2 * xm}")
    diff_res = cp.empty([height - 2 * ym, width - 2 * xm], dtype=cp.float32)

    cuda_change_map(((diff_res.shape[0] * diff_res.shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE,), 
                    (BLOCK_SIZE,),
                    (cp.asarray(im1, dtype=cp.float32), 
                     cp.asarray(im2, dtype=cp.float32),
                     diff_res,
                     width, height, xm, ym))

    if return_numpy:
        np_diff_res = cp.asnumpy(diff_res)
        return np_diff_res
    else:
        return diff_res


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


def my_dft(im, eq=False, int=False):
    eps = 1e-5
    # imf = fft2(im)
    # mag = np.abs(imf)
    imf = cv2.dft(np.float32(im),flags = cv2.DFT_COMPLEX_OUTPUT)
    mag = np.sqrt(np.square(imf[:, :, 0]) + np.square(imf[:, :, 1]))
    # imp = np.log(mag + eps)
    imp = 20 * np.log10(mag + eps) # convert to dB scale
    imp = cv2.equalizeHist(imp.astype(np.uint8)) if eq else imp
    imp = imp.astype(np.uint8) if int else imp
    return imp


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

    img1 = my_dft(img1, eq=False, int=False)
    img2 = my_dft(img2, eq=False, int=False)

    score = sk_cpt_ssim(img1, img2, multichannel=False) #输入灰度图 , multichannel=True
    return score


def gkern(height=7, width=7, std=3, scale=True):
    """Returns a 2D Gaussian kernel array."""
    gkernh = signal.gaussian(height, std=std).reshape(height, 1)
    gkernv = signal.gaussian(width, std=std).reshape(width, 1)
    gkern2d = np.outer(gkernh, gkernv)
    gkern2d = gkern2d / np.sum(gkern2d) * gkern2d.size if scale else gkern2d
    return gkern2d

def ssim_(band1, band2):
  
    K = 1e-5
    win_size = 7
    fargs = {'size': win_size}
    corr = np.abs(band1 * band2.conj())
    varr = np.abs(band1) ** 2 + np.abs(band2) ** 2
    corr_band = uniform_filter(corr, **fargs)
    varr_band = uniform_filter(varr, **fargs)
    cssim_map = (2 * corr_band + K) / (varr_band + K)
    gauss_kern = gkern(height=corr.shape[0], width=corr.shape[1], std=np.max(corr.shape) / 4)
    if len(cssim_map.shape) == 3:
        nchan = cssim_map.shape[2]
        gauss_kern = np.expand_dims(gauss_kern, axis=-1)
        gauss_kern = np.tile(gauss_kern, [1, 1, nchan])
    cssim_map = cssim_map * gauss_kern
    res = np.mean(cssim_map)
    return res

def cw_ssim_index(im1, im2, height='auto', order=4):
    """
    连续小波变换ssim。
    """
    if len(im1.shape) == 3:
        im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    if len(im2.shape) == 3:
        im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)

    assert im1.shape == im2.shape, "im1.shape == im2.shape"
    nrow, ncol = im1.shape
    pyr1 = pt.pyramids.SteerablePyramidFreq(im1, height=height, order=order, is_complex=True)
    pyr2 = pt.pyramids.SteerablePyramidFreq(im2, height=height, order=order, is_complex=True)
    level = pyr1.num_scales
    nori = pyr1.num_orientations 
    ssims = []
  
    for i in range(level):
        for j in range(nori):
            band1 = pyr1.pyr_coeffs[(i, j)]
            band2 = pyr2.pyr_coeffs[(i, j)]
            ssims.append(ssim_(band1, band2))

    res = np.mean(ssims)
    return res

def sift_create(img, rm_regs=[]):
    """
    提取sift特征
    args:
        img: 图像数据
        rm_regs: 需要屏蔽的区域， 如[[0,0,1,0.1],[0,0.9,1,1]]或者[[xmin,ymin,xmax,ymax], ..]
    return: 
        (kps, feat): (坐标点集， 特征值集)
    """
    
    ## 彩图转灰度图，灰度图是二维的
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    ## 直方图均衡化，增加图片的对比度
    # img = cv2.equalizeHist(img)
    H, W = img.shape[:2]

    ## 使用cudasift提取sift特征
    if sift_lib == "cudasift":
        numOctaves = 5;   # /* Number of octaves in Gaussian pyramid */
        initBlur = 1.6;  # /* Amount of initial Gaussian blurring in standard deviations */
        thresh = 2;    # 3.5 /* Threshold on difference of Gaussians for feature pruning */
        minScale = 0.0;  # /* Minimum acceptable scale to remove fine-scale features */
        upScale = True;  # /* Whether to upscale image before extraction */
        lmt = 43560000
        rate = 1
        if img.size > lmt: ## 若图片分辨率过大会报错，因此需要resize
            rate = math.ceil(img.size / lmt)
            img = cv2.resize(img, (int(W / rate), int(H / rate)))

        cudasift.ExtractKeypoints(img, sift_data, numOctaves, initBlur, thresh, minScale, upScale)
        df, feat = sift_data.to_data_frame()
        feat = np.ascontiguousarray(feat)
        points = zip(df['xpos'], df['ypos'])
        if rate > 1:
            kps = [cv2.KeyPoint(p[0]*rate, p[1]*rate, size=1) for p in points] ## pt还原
        else:
            kps = [cv2.KeyPoint(p[0], p[1], size=1) for p in points] ## 注意，改代码需要cv2的4.5.5版本以上

    ## 使用opencv提取sift特征
    else:
        print("Notice: sift create use cv2 !!!")
        sift = cv2.SIFT_create() # 创建sift对象
        # sift = cv2.xfeatures2d.SIFT_create()
        # kps: 关键点，包括 angle, class_id, octave, pt, response, size
        # feat: 特征值，每个特征点的特征值是128维
        kps, feat = sift.detectAndCompute(img, None) #提取sift特征

    ## 移除指定区域的特征点
    if len(rm_regs) > 0:
        if max(max(rm_regs)) <= 1:
            rm_regs = [[int(c[0]*W), int(c[1]*H), int(c[2]*W), int(c[3]*H)] for c in rm_regs]
        rm_ids = []
        for reg in rm_regs:
            for i in range(len(kps)):
                pt_ = kps[i].pt
                if reg[0] <= pt_[0] <= reg[2] and reg[1] <= pt_[1] <= reg[3]:
                    rm_ids.append(i)
        _kps = []
        for i in range(len(kps)):
            if i not in rm_ids:
                _kps.append(kps[i])
        kps = _kps
        feat = np.delete(feat, rm_ids, axis=0)
    
    # ref_sift = cv2.drawKeypoints(ref_img,kps1,ref_img,color=(255,0,255)) # 画sift点
    # cv2.imwrite("images/test_sift.jpg", hmerge)

    return (kps, feat)


def sift_match(feat_ref, feat_tag, ratio=0.5, ops="Affine"):
    """
    使用sift特征，flann算法计算两张轻微变换的图片的的偏移转换矩阵M。
    args:
        feat_ref: 参考图片的sift特征，格式为：(kps, feat)
        feat_tag: 待分析图片的sift特征，格式为：(kps, feat)
        ratio: sift点正匹配的阈值
        ops: 变换的方式，可选择"Affine"(仿射), "Perspective"(投影)
    return:
        M:偏移矩阵, 2*3矩阵，前两列表示仿射变换，后一列表示平移量。
          偏移后的点的计算公式：(x', y') = M * (x, y, 1)
    """
    kps1, feat1 = feat_ref
    kps2, feat2 = feat_tag

    if  feat1 is None or feat2 is None or len(feat1) < 3 or len(feat2) < 3:
        print("warning: img have no sift feat!")
        return None
    
    ## flann 快速最近邻搜索算法，计算两张特征的正确匹配点。
    ## https://www.cnblogs.com/shuimuqingyang/p/14789534.html
    try:
        ## 使用gpu计算sift matches
        feat1_gpu = cv2.cuda_GpuMat()
        feat1_gpu.upload(feat1) # 将数据转为cuda形式
        feat2_gpu = cv2.cuda_GpuMat()
        feat2_gpu.upload(feat2)
        matcherGPU = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_L2)
        matches = matcherGPU.knnMatch(feat1_gpu, feat2_gpu, k=2)
    except:
        # bf = cv2.BFMatcher()
        # matches = bf.knnMatch(feat1, feat2, k=2)
        print("Warning: sift match with cpu !!")
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
    # print("num of good match pointer:", len(good))

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

        # print("Enough matches are found - {}/{}".format(len(good), min_match_count))
        return M
    else:
        # print("Not enough matches are found - {}/{}".format(len(good), min_match_count))
        return None


def correct_offset(tag_img, M, b=False):
    """
    根据偏移矩阵矫正图片。
    args:
        tag_img: 待矫正图片
        M: 偏移矩阵
        b: bool， 是否返回黑边范围
    return:
        if b == False:
            img_tag_warped: 矫正之后的tag_img
        else:
            (img_tag_warped, [xmin, ymin, xmax, ymax])
    """
    if M is None:
        if b:
            return tag_img, [0,0,tag_img.shape[1], tag_img.shape[0]]
        else:
            return tag_img
    
    ## 矫正图片
    if M.shape == (2, 3):  # warp affine
        img_tag_warped = cv2.warpAffine(tag_img, M, (tag_img.shape[1], tag_img.shape[0]), flags=cv2.WARP_INVERSE_MAP)
    elif M.shape == (3, 3):  # warp perspective
        img_tag_warped = cv2.warpPerspective(tag_img, M, (tag_img.shape[1], tag_img.shape[0]), flags=cv2.WARP_INVERSE_MAP)

    if b:
        Mi = cv2.invertAffineTransform(M)
        x0, y0 = 0, 0
        x2, y2 = img_tag_warped.shape[1] - 1, img_tag_warped.shape[0] - 1
        x1, y1 = x0, y2
        x3, y3 = x2, y0
        points = np.stack([[[x0, y0]],[[x1, y1]],[[x2, y2]],[[x3, y3]]])
        res = cv2.transform(points, Mi)
        (x0t, y0t), (x1t, y1t), (x2t, y2t), (x3t, y3t) = np.squeeze(res, 1)
        xmin = max(x0, x0t, x1t)
        xmax = min(x2, x2t, x3t)
        ymin = max(y0, y0t, y3t)
        ymax = min(y2, y2t, y1t)
        
        return img_tag_warped, [xmin, ymin, xmax, ymax]

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

def find_vector_set(diff_image, new_size):
    """
    主成分分析
    """
    i = 0
    j = 0
    vector_set = np.zeros((int(new_size[0] * new_size[1] / 25), 25))

    for j in range(0, new_size[0], 5):
        for k in range(0, new_size[1], 5):
            block = diff_image[j:j+5, k:k+5]
            #print(i,j,k,block.shape)
            feature = block.ravel()
            vector_set[i, :] = feature
            i += 1
        
    mean_vec   = np.mean(vector_set, axis = 0)    
    vector_set = vector_set - mean_vec
    
    return vector_set, mean_vec

def find_FVS(EVS, pca, diff_image, mean_vec, new):
    ## 求FVS, 使用cython的版本
    if vector_lib == "cython":
        diff_image = diff_image.astype(np.uint8)
        feature_vector_set = pack_vector_set(diff_image, np.array(new))
        feature_vector_set = np.array(feature_vector_set, dtype=np.float64)

    ## FVS的完整计算过程
    else:
        print("Notice: vector_set use python !!!")
        i = 2 
        feature_vector_set = []
        
        while i < new[0] - 2:
            j = 2
            while j < new[1] - 2:
                block = diff_image[i-2:i+3, j-2:j+3]
                feature = block.flatten()
                feature_vector_set.append(feature)
                j = j+1
            i = i+1
    FVS = pca.transform(feature_vector_set)
    return FVS

def clustering(FVS, new):
    if kmeans_lib == "sklearn":
        print("Notice: PCA and KMeans use sklearn !!!")
    if kmeans_lib == "cuml":
        FVS = FVS.astype(np.float32)
        FVS = cudf.DataFrame(FVS)
    kmeans.fit(FVS)
    output = kmeans.predict(FVS)
    if kmeans_lib == "cuml":
        output = output.to_numpy()
    count  = Counter(output)

    least_index = min(count, key = count.get)            
    change_map  = np.reshape(output,(new[0] - 4, new[1] - 4))
    
    return least_index, change_map

def pca_kmeans(dif_img):
    """
    对图片做pca-kmeans，获得二值图
    https://appliedmachinelearning.blog/2017/11/25/unsupervised-changed-detection-in-multi-temporal-satellite-images-using-pca-k-means-python-code/
    args:
        dif_img: img_data
    return:
        dif_img:pca-kmeans后的二值图
    """
    if len(dif_img.shape) == 3:
        dif_img = cv2.cvtColor(dif_img, cv2.COLOR_RGB2GRAY)
    img_size= dif_img.shape
    vector_set, mean_vec = find_vector_set(dif_img, img_size)
    pca = PCA(n_components=25) # n_components=25
    pca.fit(vector_set)
    EVS = pca.components_
    FVS = find_FVS(EVS, pca, dif_img, mean_vec, img_size)
    least_index, dif_img = clustering(FVS, img_size)
    dif_img[dif_img == least_index] = 255
    dif_img[dif_img != 255] = 0
    dif_img = dif_img.astype(np.uint8)
    return dif_img

def process_binary_img(dif_img):
    """
    对二值图做后处理，例如，去除零星的点
    args:
        dif_img: 二值图
    return:
        dif_img: 处理后的二值图
    """
    ## 对差异性图片进行腐蚀操作，去除零星的点。
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,5))
    kernel  = np.asarray(((0,0,1,0,0),(0,1,1,1,0),(1,1,1,1,1),(0,1,1,1,0),(0,0,1,0,0)), dtype=np.uint8)
    dif_img = cv2.erode(dif_img,kernel,iterations=1)
    # cv2.imwrite("test1/tag_diff_erode.jpg",dif_img)

    ## 去除轮廓面积小的点
    contours, hierarchy = cv2.findContours(dif_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.array(contours)
    areas = [cv2.contourArea(c) for c in contours]
    if len(areas) < 1:
        return dif_img
    pts = [cv2.boundingRect(c)[:2] for c in contours]
    ind_max_area = areas.index(max(areas))
    max_area = areas[ind_max_area]
    if max_area < 2:
        return dif_img
    max_pt = pts[ind_max_area]

    d_max = math.sqrt(dif_img.shape[0] ** 2 + dif_img.shape[1] ** 2)
    at = [0, 0.2] ## 面积范围
    t_inds = [] ## 最终符合面积要求的contour的index
    for i in range(len(areas)):
        d = math.sqrt((max_pt[0]-pts[i][0]) ** 2 + (max_pt[1]-pts[i][1]) ** 2)
        if areas[i] / max_area > at[0] + (at[1] - at[0]) * ((d / d_max) ** 2):
            t_inds.append(i)

    dif_img = cv2.drawContours(np.zeros_like(dif_img), contours[t_inds], -1, 255, -1)
    return dif_img
    

def detect_diff(img_ref, img_tag):
    """
    判别算法，检测出待分析图与基准图的差异区域
    return:
        rec_real: 差异区域，若判定没差异，则返回[]
    """
    img_tag_ = img_tag.copy()
    ## 将img_tag_hsv的亮度(v)平均值调整为等于img_ref_hsv的亮度(v)平均值
    # img_ref_hsv = cv2.cvtColor(img_ref, cv2.COLOR_BGR2HSV)
    # img_tag_hsv = cv2.cvtColor(img_tag, cv2.COLOR_BGR2HSV)
    # rate = np.sum(img_ref_hsv[:, :, 2]) / np.sum(img_tag_hsv[:, :, 2])
    # img_tag_hsv[:, :, 2] = rate * img_tag_hsv[:, :, 2]
    # img_tag = cv2.cvtColor(img_tag_hsv, cv2.COLOR_HSV2BGR)

    ## 将图片转为灰度图后相减，得到差异图片
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY)
    if len(img_tag.shape) == 3:
        img_tag = cv2.cvtColor(img_tag, cv2.COLOR_RGB2GRAY)
    r = 1
    maxlen = 500
    raw_size = img_tag.shape[:2]
    if max(raw_size) > maxlen:
        r = maxlen / max(raw_size)
    img_size = (np.asarray(img_tag.shape) * r / 5).astype(np.int) * 5
    img_tag = cv2.resize(img_tag, (img_size[1], img_size[0]))
    img_ref = cv2.resize(img_ref, (img_size[1], img_size[0]))

    (score,dif_img) = sk_cpt_ssim(img_tag,img_ref,full = True)
    print("ssim between img_tag and img_ref is:", score)
    if score > 0.995:
        return []
    
    ## 像素相减获取差异图片
    # img_ref = exposure.match_histograms(img_ref, img_tag).astype(np.uint8) ## 
    # dif_img = img_tag.astype(float) - img_ref.astype(float) ## 直接相减的差异
    # dif_img = np.abs(dif_img).astype(np.uint8)
    tol = 0.01
    dif_img = get_change_map(img_ref, img_tag, tol=tol) ## 周围像素相减，取最小值
    dif_img = cv2.resize(dif_img, (img_size[1], img_size[0]))
    dif_img = dif_img.astype(np.uint8)
    # cv2.imwrite("test1/tag_diff.jpg", dif_img)

    ## 对差异图进行二值化
    dif_img = pca_kmeans(dif_img) # pca-kmeans求二值图
    # _, dif_img = cv2.threshold(dif_img, 80, 255, cv2.THRESH_BINARY) # 二值化
    # cv2.imwrite("test1/tag_diff_thre.jpg",dif_img)

    ## 对二值图做后处理，去除零星的点
    dif_img = process_binary_img(dif_img)
    # cv2.imwrite("test1/tag_diff_thre.jpg",dif_img)

    ## 用最小外接矩阵框出差异的地方
    
    index_255 = np.where(dif_img == 255)
    index_255 = [a for a in index_255 if len(a) > 1]
    if len(index_255) > 1:
        ymin = max(0, min(index_255[0])-3)
        xmin = max(0, min(index_255[1])-3)
        ymax = min(dif_img.shape[0], max(index_255[0])+3)
        xmax = min(dif_img.shape[1], max(index_255[1])+3)
        rec_dif = [xmin, ymin, xmax, ymax]
    else:
        rec_dif = []
    ## 若最终框面积不在0.1 - 0.5 之间，返回空。
    H, W = dif_img.shape
    if len(rec_dif) > 1:
        dif_area = (rec_dif[2] - rec_dif[0]) * (rec_dif[3] - rec_dif[1])
        if dif_area / (H * W) > 0.5:
            rec_dif = []

    ## 将矩形框还原回原始大小
    if len(rec_dif) > 1:
        r = img_tag_.shape[0] / dif_img.shape[0]
        rec_dif = [int(r*rec_dif[0]), int(r*rec_dif[1]), int(r*rec_dif[2]), int(r*rec_dif[3])]
    # cv2.rectangle(img_tag_, (rec_dif[0], rec_dif[1]), (rec_dif[2], rec_dif[3]), (0,0,255), 2)
    # cv2.imwrite("test1/tag_diff_rec.jpg",img_tag_)

    return rec_dif

if __name__ == '__main__':
    import os
    import glob

    ref_file = "test1/0996_normal.jpg"
    tag_file = "test1/0996_3.jpg"
    img_ref = cv2.imread(ref_file) 
    img_tag = cv2.imread(tag_file)

    # resize, 降低分别率，加快特征提取的速度。
    resize_max = 1280
    H, W = img_ref.shape[:2]
    resize_rate = max(H, W) / resize_max  ## 缩放倍数
    img_ref = cv2.resize(img_ref, (int(W / resize_rate), int(H / resize_rate)))
    H, W = img_tag.shape[:2]  ## resize
    img_tag = cv2.resize(img_tag, (int(W / resize_rate), int(H / resize_rate)))

    feat_ref = sift_create(img_ref)
    feat_tag = sift_create(img_tag) # 提取sift特征
    M = sift_match(feat_tag, feat_ref, ratio=0.5, ops="Affine")

    img_ref, cut = correct_offset(img_ref, M, b=True)
    img_tag = img_tag[cut[1]:cut[3], cut[0]:cut[2], :]
    img_ref = img_ref[cut[1]:cut[3], cut[0]:cut[2], :]
    diff_area = detect_diff(img_ref, img_tag)

    # rec_real = [rec_dif[0] + c[0], rec_dif[1] + c[1],rec_dif[2] + c[0],rec_dif[3] + c[1]]

    