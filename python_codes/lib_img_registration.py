import cv2

import numpy as np
import torch
import math
from imreg import similarity

try:
    import cupy as cp ## pip install cupy-cuda114
    is_cupy = True
except:
    is_cupy = False
    print("warning: no cupy pkg !")

try:
    import kornia as K # pip install kornia
    import kornia.feature as KF
except:
    print("warning: no kornia pkg !")

try:
    from lib_lightglue_onnx import lightglue_registration_onnx, lightglue_registration_om
    registration_opt = "lightglue"
except:
    print("Warning: Not gpu memory enough to load lightglue module !")
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

    # ## 使用cupy打包array
    if is_cupy:
        img_tag = cp.array(img_tag)
        ref_resize = cp.array(ref_resize)

    ## 计算偏移矩阵
    im_warped, scale, angle, (t0, t1) = similarity(img_tag, ref_resize, retained_angle) 
    if is_cupy:
        scale, angle, t0, t1 = 1 / float(scale), float(angle.get()), t0.get(), t1.get()
    else:
        scale, angle, t0, t1 = 1 / float(scale), float(angle), t0, t1
    tx, ty = -t1, -t0
    center = img_tag.shape[1] // 2, img_ref.shape[0] // 2
    M = getAffine(center, angle, scale, (tx, ty))

    ## 若resize过，则将resize信息放入到M中
    M = np.dot(M, M_scale)

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
    if registration_opt == "lightglue":
        # return lightglue_registration_onnx(img_ref, img_tag)
        return lightglue_registration_om(img_ref, img_tag)

    if registration_opt == "fft":
        print("registration with fft !")
        return fft_registration(img_ref, img_tag)
    
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
        xmin = max(x0, min(x0t, x1t, x2t, x3t))
        xmax = min(x2, max(x0t, x1t, x2t, x3t))
        ymin = max(y0, min(y0t, y1t, y2t, y3t))
        ymax = min(y2, max(y0t, y1t, y2t, y3t))
        
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

def roi_registration(img_ref, img_tag, roi_ref):
    """
    roi框纠偏，将img_ref上的roi框纠偏匹配到img_tag上
    args:
        roi_ref: roi字典，{"roi_1": [xmin, ymin, xmax, ymax]}
    return:
        roi_tag: 纠偏后的roi框, {"roi_1": [xmin, ymin, xmax, ymax]}
    """
    H, W = img_tag.shape[:2]
    if len(roi_ref) == 0:
        roi_ref = {"no_roi": [0,0,W,H]}
    
    if img_ref is None:
        return roi_ref, None

    M = registration(img_ref, img_tag) # 求偏移矩阵

    if M is None:
        return roi_ref, None
    
    roi_tag = {}
    for name in roi_ref:
        roi = roi_ref[name]
        coors = [(roi[0],roi[1]), (roi[2],roi[1]), (roi[2],roi[3]), (roi[0],roi[3])]
        coors_ = [list(convert_coor(coor, M)) for coor in coors]
        c_ = np.array(coors_, dtype=int)
        r = [min(c_[:,0]), min(c_[:, 1]), max(c_[:,0]), max(c_[:,1])]
        r = [int(r_) for r_ in r]
        roi_tag[name] = [max(0, r[0]), max(0, r[1]), min(W, r[2]), min(H, r[3])]

    return roi_tag, M

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
