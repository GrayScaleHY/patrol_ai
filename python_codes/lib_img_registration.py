import cv2
import kornia as K # pip install kornia
import kornia.feature as KF
import numpy as np
import torch
import math
import cupy as cp ## pip install cupy-cuda114
from imreg import similarity

try:
    matcher = KF.LoFTR(pretrained='outdoor').cuda().half()
    registration_opt = "loftr"
except:
    print("Warning: Not gpu memory enough to load LoFTR module !")
    registration_opt = "fft"


def getAffine(center, angle, scale, trans):             
    R = cv2.getRotationMatrix2D(center, angle, scale)
    R = np.vstack((R, np.array([0, 0, 1])))                         
    T = np.array([[1., 0., trans[0]], [0., 1., trans[1]]])                         
    M = T @ R                                                       
    M = M[:2]  
    return M 

def fft_registration(img_ref, img_tag, retained_angle=45):
    """
    基于fft的图像矫正方法
    https://github.com/zldrobit/imreg/tree/opencv
    args:
        img_ref: 参考图
        img_tag: 待矫正图
        retained_angle: 旋转角度限制
    return:
        M: 偏移矩阵, 2*3矩阵，偏移后的点的计算公式：(x', y') = M * (x, y, 1)
    """
    if len(img_tag.shape) == 3:
        img_tag = cv2.cvtColor(img_tag, cv2.COLOR_RGB2GRAY)
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY)
    
    ## 判断img_tag和img_ref是否分辨率一样，若不一样则需要resize.
    h_r, w_r = img_ref.shape[:2]
    h_t, w_t = img_tag.shape[:2]
    if h_r != h_t or w_r != w_t:
        resize_scale = [w_t / w_r, h_t / h_r, 1]
        ref_resize = cv2.resize(img_ref, (w_t, h_t))
    else:
        resize_scale = [1 ,1, 1]
        ref_resize = img_ref
    M_scale = np.diag(np.array(resize_scale)) 

    ## 使用cupy打包array
    img_tag = cp.array(img_tag)
    ref_resize = cp.array(ref_resize)

    ## 计算偏移矩阵
    im_warped, scale, angle, (t0, t1) = similarity(img_tag, ref_resize, retained_angle) 
    scale, angle, t0, t1 = 1 / float(scale), float(angle.get()), t0.get(), t1.get()
    tx, ty = -t1, -t0
    center = img_tag.shape[1] // 2, img_ref.shape[0] // 2
    M = getAffine(center, angle, scale, (tx, ty))

    ## 若resize过，则将resize信息放入到M中
    M = np.dot(M, M_scale)

    return M

def loftr_registration(img_ref, img_tag, max_size=1280):
    """
    基于transformer的局部特征匹配纠偏算法
    https://blog.csdn.net/weixin_48109360/article/details/121422277
    args:
        img_ref: 参考图
        img_tag: 待矫正图
        max_size: 图片最长边限制
    return:
        M: 偏移矩阵, 2*3矩阵，偏移后的点的计算公式：(x', y') = M * (x, y, 1)
    """
    # 若图片最长边大于max_size，将图片resize到max_size内
    shape_max = max(list(img_ref.shape[:2]) + list(img_tag.shape[:2]))
    resize_rate = math.ceil(shape_max / max_size)
    H, W = img_tag.shape[:2]
    img_tag = cv2.resize(img_tag, (int(W / resize_rate), int(H / resize_rate)))
    H, W = img_ref.shape[:2]
    img_ref = cv2.resize(img_ref, (int(W / resize_rate), int(H / resize_rate)))

    resize_scale = [1, 1, resize_rate] # 原始图片相对于resize后的图片的偏移矩阵M的关系。
    M_scale = np.diag(np.array(resize_scale)) 

    # 将numpy转为tensor
    img_tag = K.image_to_tensor(img_tag, False).cuda().float().half() /255.
    img_tag = K.color.bgr_to_rgb(img_tag)
    img_ref = K.image_to_tensor(img_ref, False).cuda().float().half() /255.
    img_ref = K.color.bgr_to_rgb(img_ref)

    # LofTR works on grayscale images only 
    input_dict = {"image0": K.color.rgb_to_grayscale(img_ref), 
              "image1": K.color.rgb_to_grayscale(img_tag)}
    with torch.inference_mode():
        matches = matcher(input_dict)
    src_pts = matches['keypoints0'].float()
    dst_pts = matches['keypoints1'].float()

    # 特征匹配
    cpu_src_pts, cpu_dst_pts = src_pts.cpu().numpy(), dst_pts.cpu().numpy()  # take into account gpu -> cpu time
    M, mask = cv2.estimateAffine2D(cpu_src_pts, cpu_dst_pts)

    M = np.dot(M, M_scale) # 偏移矩阵还原会原始图片对应的M

    return M

def registration(img_ref, img_tag):
    """
    偏移矫正算法整合
    args:
        img_ref: 参考图
        img_tag: 待矫正图
    return:
        M: 偏移矩阵, 2*3矩阵，偏移后的点的计算公式：(x', y') = M * (x, y, 1)
    """
    if registration_opt == "fft":
        print("registration with fft !")
        return fft_registration(img_ref, img_tag)
    
    try:
        print("registration with LoFTR !")
        return loftr_registration(img_ref, img_tag) # 优先使用Loftr纠偏算法
    except:
        print("registration with fft !")
        return fft_registration(img_ref, img_tag) # 否则使用fft纠偏算法


if __name__ == '__main__':
    import time
    from lib_sift_match import convert_coor
    tag_file = "/data/PatrolAi/result_patrol/pointer/0322140730_1号主变A相东侧油温_tag.jpg"
    ref_file = "/data/PatrolAi/result_patrol/pointer/0322140730_1号主变A相东侧油温_ref.jpg"

    coor = []

    img_tag = cv2.imread(tag_file)
    img_ref = cv2.imread(ref_file)
    H, W = img_tag.shape[:2]
    print("W:", W, "H:", H)
    # img_tag = cv2.resize(img_tag, (int(W / 2), int(H / 2)))
    img_ref = cv2.resize(img_ref, (int(W / 2), int(H / 3)))

    start = time.time()
    M = loftr_registration(img_ref, img_tag)
    print("all loftr spend time:", time.time() - start)
    print("M:", M)

    # seg = [400, 100, 500, 200]
    # cv2.line(img_ref, (int(seg[0]), int(seg[1])),
    #              (int(seg[2]), int(seg[3])), (0, 255, 0), 2)
    # seg[:2] = convert_coor(seg[:2], M)
    # seg[2:4] = convert_coor(seg[2:4], M)
    # cv2.line(img_tag, (int(seg[0]), int(seg[1])),
    #              (int(seg[2]), int(seg[3])), (0, 255, 0), 2)
    # cv2.imwrite("img_tag.jpg", img_tag)
    # cv2.imwrite("img_ref.jpg", img_ref)
    