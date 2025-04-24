import cv2

import numpy as np
import torch
import math
from config_model_list import registration_model_list


# try:
#     import sys
#     sys.path.insert(0,'../SuperGluePretrainedNetwork')
#     from models.matching import Matching
#     from models.utils import frame2tensor
#     torch.set_grad_enabled(False)
#     device='cuda'
#     config = {'superpoint': {'nms_radius': 4,'keypoint_threshold': 0.005,'max_keypoints': -1},
#                 'superglue': {'weights': "indoor",'sinkhorn_iterations': 20,'match_threshold': 0.6}}
#     matching = Matching(config).eval().to(device) # 初始化模型
#     registration_opt = "superglue"
# except:
#     from imreg import similarity

# try:
#     import cupy as cp ## pip install cupy-cuda114
#     registration_opt = "fft"
#     is_cupy = True
# except:
#     is_cupy = False
#     print("Warning: Not gpu memory enough to load LoFTR module !")
#     registration_opt = "fft"

def eflotr_choice_set():
    from src.loftr import LoFTR, full_default_cfg, reparameter
    from copy import deepcopy

    _default_cfg = deepcopy(full_default_cfg)
    weights = "/checkpoint/eloftr_outdoor.ckpt"
    matcher = LoFTR(config=_default_cfg)
    matcher.load_state_dict(torch.load(weights)['state_dict'])
    matcher = reparameter(matcher)  # no reparameterization will lead to low performance
    matcher = matcher.half().eval().to("cuda")
    registration_opt = "eflotr"
    extractor=None
    device=None
    return matcher,registration_opt,extractor,device

def lightglue_choice_set():
    '''
    https://github.com/cvg/LightGlue/tree/main
    python -m pip install -e .
    '''
    from lightglue import LightGlue, SuperPoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cuda")
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
    matcher = LightGlue(features="superpoint").eval().to(device)
    registration_opt = "lightglue"
    return matcher,registration_opt,extractor,device

def loftr_choice_set():
    import kornia.feature as KF
    print("warning: no lightglue pkg !")
    matcher = KF.LoFTR(pretrained='outdoor').cuda().half()
    registration_opt = "loftr"
    extractor=None
    device=None
    return matcher, registration_opt, extractor,device


try:
    matcher,registration_opt,extractor,device=eval(registration_model_list[1]+"_choice_set")()
except:
    try:
        matcher,registration_opt,extractor,device=eval(registration_model_list[2]+"_choice_set")()
    except:
        matcher,registration_opt,extractor,device=eval(registration_model_list[3]+"_choice_set")()


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)

def lightglue_registration(img_ref, img_tag, max_size=1280):
    """
    lightglue纠偏算法
    https://github.com/cvg/LightGlue/tree/main
    https://colab.research.google.com/github/cvg/LightGlue/blob/main/demo.ipynb#scrollTo=6JA4sWG9PV7M
    """
    # 若图片最长边大于max_size，将图片resize到max_size内
    from lightglue.utils import rbd
    shape_max = max(list(img_ref.shape[:2]) + list(img_tag.shape[:2]))
    resize_rate = math.ceil(shape_max / max_size)
    H, W = img_tag.shape[:2]
    img_tag = cv2.resize(img_tag, (int(W / resize_rate), int(H / resize_rate)))
    H, W = img_ref.shape[:2]
    img_ref = cv2.resize(img_ref, (int(W / resize_rate), int(H / resize_rate)))
    resize_scale = [1, 1, resize_rate] # 原始图片相对于resize后的图片的偏移矩阵M的关系。
    M_scale = np.diag(np.array(resize_scale)) 

    img_ref = numpy_image_to_torch(img_ref)
    img_tag = numpy_image_to_torch(img_tag)

    feats0 = extractor.extract(img_ref.to(device))
    feats1 = extractor.extract(img_tag.to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})

    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    mkpts0, mkpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    mkpts0 = mkpts0.cpu().numpy()
    mkpts1 = mkpts1.cpu().numpy()
    if len(mkpts0) < 3 or len(mkpts1) < 3:
        return None
    M, mask = cv2.estimateAffinePartial2D(mkpts0, mkpts1, method=cv2.RANSAC, ransacReprojThreshold=5)
    # M, mask = cv2.estimate(mkpts0, mkpts1, method=cv2.RANSAC, ransacReprojThreshold=5)
    # M, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 8)

    if M is None:
        return None

    M = np.dot(M, M_scale) # 偏移矩阵还原会原始图片对应的M
    
    return M

def superglue_registration(img_ref, img_tag):
    """
    基于superpoint+superglue的图像配准算法
    https://github.com/magicleap/SuperGluePretrainedNetwork/tree/master
    """

    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY)
    if len(img_tag.shape) == 3:
        img_tag = cv2.cvtColor(img_tag, cv2.COLOR_RGB2GRAY)

    keys = ['keypoints', 'scores', 'descriptors']
    frame_tensor = frame2tensor(img_ref, device)
    last_data = matching.superpoint({'image': frame_tensor})
    last_data = {k+'0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor

    frame_tensor = frame2tensor(img_tag, device)
    pred = matching({**last_data, 'image1': frame_tensor})
    kpts0 = last_data['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    # confidence = pred['matching_scores0'][0].cpu().numpy()
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    # confidence = confidence[valid]

    if len(mkpts0) < 10:
        return None

    M, mask = cv2.estimateAffinePartial2D(mkpts0, mkpts1, method=cv2.RANSAC, ransacReprojThreshold=5)
    # M, _ = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 8)
    # M, mask = cv2.estimateAffine2D(mkpts0, mkpts1)
    return M

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
    import kornia as K  # pip install kornia
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


def eloftr_resize(coor, ref_shape, tag_shape, max_size=1280):
    """
        针对eloftr存在32倍数偏差问题
        将目标坐标点随着图像进行缩放
        args:
            coor: 输入坐标, (x, y) or [x, y]
            ref_shape : ref图像的尺寸, (H, W)
            tag_shape : tag图像的尺寸, (H, W)
        return:
            coor: 纠正缩放后坐标, (x, y)
    """
    shape_max = max(list(ref_shape) + list(tag_shape))
    resize_rate = math.ceil(shape_max / max_size)
    H_ref, W_ref = ref_shape
    new_H = int(H_ref / resize_rate)//32 * 32
    new_W = int(W_ref / resize_rate)//32 * 32
    # 除以ref缩放因子
    coor = (coor[0] / W_ref * new_W, coor[1] / H_ref * new_H)

    H_tag, W_tag = tag_shape
    new_H = int(H_tag / resize_rate)//32 * 32
    new_W = int(W_tag / resize_rate)//32 * 32
    # 乘上tag缩放因子
    coor = (coor[0] * W_tag / new_W, coor[1] *H_tag / new_H)

    return coor


def eloftr_registration(img_ref, img_tag, max_size=1280):
    """
    ELoFTR: LoFTR升级版, cvpr2024论文EfficientLoFTR
    args:
        img_ref: 参考图
        img_tag: 待矫正图
    return:
        M: 偏移矩阵, 2*3矩阵，偏移后的点的计算公式：(x', y') = M * (x, y, 1)
    """
    img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    img_tag = cv2.cvtColor(img_tag, cv2.COLOR_BGR2GRAY)

    # 若图片最长边大于max_size，将图片resize到max_size内
    shape_max = max(list(img_ref.shape[:2]) + list(img_tag.shape[:2]))
    resize_rate = math.ceil(shape_max / max_size)
    
    H, W = img_tag.shape[:2]
    new_H = int(H / resize_rate)//32 * 32
    new_W = int(W / resize_rate)//32 * 32
    img_tag = cv2.resize(img_tag, (new_W, new_H))

    H, W = img_ref.shape[:2]
    new_H = int(H / resize_rate)//32 * 32
    new_W = int(W / resize_rate)//32 * 32
    img_ref = cv2.resize(img_ref, (new_W, new_H))

    resize_rate_H = H / new_H
    resize_rate_W = W / new_W
    M_scale = np.array([[1, 1, resize_rate_W], [1, 1, resize_rate_H]])

    img0 = torch.from_numpy(img_ref)[None][None].half().to("cuda") / 255.
    img1 = torch.from_numpy(img_tag)[None][None].half().to("cuda") / 255.


    batch = {'image0': img0, 'image1': img1}

    # Inference with EfficientLoFTR and get prediction
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        # mconf = batch['mconf'].cpu().numpy()

    M, mask = cv2.estimateAffine2D(mkpts0, mkpts1)
    # M = np.dot(M, M_scale)
    M *= M_scale

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
    if registration_opt == "eflotr":
        print("registration with eloftr !")
        return eloftr_registration(img_ref, img_tag) # 使用eloftr纠偏算法
    if registration_opt == "loftr":
        print("registration with LoFTR !")
        return loftr_registration(img_ref, img_tag) # 使用Loftr纠偏算法
    elif registration_opt == "lightglue":
        print("registration with lightglue !")
        return lightglue_registration(img_ref, img_tag) # 使用lightglue纠偏算法
    elif registration_opt == "superglue":
        print("registration with SuperGlue !")
        return superglue_registration(img_ref, img_tag) # 使用superglue纠偏算法
    else:
        print("registration with fft !")
        return fft_registration(img_ref, img_tag) # 最不优先使用fft纠偏算法
    
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
        if img_ref is not None:
            roi_ref = {"no_roi": [0,0,img_ref.shape[1],img_ref.shape[0]]}
        else:
            roi_ref = {"no_roi": [0, 0, W, H]}
            return roi_ref, None
    
    if img_ref is None:
        return roi_ref, None

    M = registration(img_ref, img_tag) # 求偏移矩阵

    if M is None:
        return roi_ref, None
    
    roi_tag = {}
    for name in roi_ref:
        roi = roi_ref[name]
        coors = [(roi[0],roi[1]), (roi[2],roi[1]), (roi[2],roi[3]), (roi[0],roi[3])]
        if registration_opt == "eflotr":
            # 纠正eloftr 32倍数偏移
            coors =  [list(eloftr_resize(coor, img_ref.shape[:2], img_tag.shape[:2])) for coor in coors]
        coors_ = [list(convert_coor(coor, M)) for coor in coors]
        c_ = np.array(coors_, dtype=int)
        r = [min(c_[:,0]), min(c_[:, 1]), max(c_[:,0]), max(c_[:,1])]
        r = [int(r_) for r_ in r]
        roi_tag[name] = [max(0, r[0]), max(0, r[1]), min(W, r[2]), min(H, r[3])]
        roi_ = roi_tag[name]
        if roi_[2] <= roi_[0] or roi_[3] <= roi_[1]:
            roi_tag[name] = roi_ref[name]
            M = None
        
    return roi_tag, M

if __name__ == '__main__':
    ref_file = "images/ref.jpg"
    tag_file = "images/tag.jpg"

    img_ref = cv2.imread(ref_file)
    img_tag = cv2.imread(tag_file)

    ## 根据图片匹配算法求两张图片之间的偏移矩阵
    M = registration(img_ref, img_tag)

    print("偏移矩阵M", M)

    ## 将ref图上的目标框匹配到tag图上

    box_ref = [23,825,120,878]

    # 将box_ref叠加到ref图上
    cv2.rectangle(img_ref, (int(box_ref[0]), int(box_ref[1])),
                        (int(box_ref[2]), int(box_ref[3])), (0, 255, 0), 2)
    cv2.imwrite(ref_file[:-4] + "_box.jpg", img_ref)

    # 求出box_ref框对应在tag图上的box_tag框，并叠加显示
    box_tag = list(convert_coor(box_ref[:2], M)) + list(convert_coor(box_ref[2:4], M))
    cv2.rectangle(img_tag, (int(box_tag[0]), int(box_tag[1])),
                        (int(box_tag[2]), int(box_tag[3])), (0, 255, 0), 2)
    cv2.imwrite(tag_file[:-4] + "_box.jpg", img_tag)
    
