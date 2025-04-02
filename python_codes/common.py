import cv2, os, requests
import numpy as np
# import onnxruntime as ort
from typing import List, Optional, Union
import math, torch
import numpy as np
from scipy.optimize import curve_fit
import base64
from datetime import datetime
import json
import time
import logging
import random


# device = torch.device("cuda")
# extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
# matcher = LightGlue(features="superpoint").eval().to(device)

def convert_pt(x, range_x):
    xmax = range_x[1]
    if range_x[0] < range_x[1]:
        xmin = range_x[0]
    else:
        xmin = range_x[0] - 360
    
    if range_x[0] == 0 and range_x[1] == 360:
        if x < 0:
            x = 360 + x

    if x > xmax:
        return xmax
    
    if x < xmin:
        x = xmin
    
    if x < 0:
        x = x + 360
    
    return x

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

def get_img(img_str):
    """
    将字符串形式的图片转成numpy图片，字符串形式图片可以是base64格式、http链接、图片绝对路径
    """
    if img_str.startswith("http"): # 如果是网络图，则尝试下载
        img_str = img_str.replace('#', '%23')
        port = ':' + img_str.split(":")[2].split("/")[0]
        # img_str_file形如/export/constast_pic1735525243759.jpg
        img_str_file = "/export" + img_str.split(port)[1]
        img_str_dir = os.path.dirname(img_str_file)
        os.makedirs(img_str_dir, exist_ok=True)
        r = requests.get(img_str)
        f = open(img_str_file, "wb")
        f.write(r.content)
        f.close()
        img = cv2.imread(img_str_file)
    elif os.path.exists(img_str): # 如果是绝对路径，则直接读取
        img = cv2.imread(img_str)

    else: # 若是base64,转换为img
        try:
            img = base642img(img_str)
        except:
            print('data can not convert to img!')
            img = None
    return img

def read_image(path: str, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale"""
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(path, mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return image / 255.0


def resize_image(
    image: np.ndarray,
    size: Union[List[int], int],
    fn: str,
    interp: Optional[str] = "area",
) -> np.ndarray:
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]

    fn = {"max": max, "min": min}[fn]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        scale = (w_new / w, h_new / h)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f"Incorrect new size: {size}")
    mode = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    }[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale


def load_image(
    path: str,
    grayscale: bool = False,
    resize: int = None,
    fn: str = "max",
    interp: str = "area",
):
    img = read_image(path, grayscale=grayscale)
    scales = [1, 1]
    if resize is not None:
        img, scales = resize_image(img, resize, fn=fn, interp=interp)
    return normalize_image(img)[None].astype(np.float32), np.asarray(scales)


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale."""
    scale = np.array([0.299, 0.587, 0.114], dtype=image.dtype).reshape(3, 1, 1)
    image = (image * scale).sum(axis=-3, keepdims=True)
    return image

def post_process(kpts0, kpts1, matches, scales0, scales1):
    kpts0 = (kpts0 + 0.5) / scales0 - 0.5
    kpts1 = (kpts1 + 0.5) / scales1 - 0.5
    # create match indices
    m_kpts0, m_kpts1 = kpts0[0][matches[..., 0]], kpts1[0][matches[..., 1]]
    return m_kpts0, m_kpts1

def normalize_keypoints(kpts: np.ndarray, h: int, w: int) -> np.ndarray:
    size = np.array([w, h])
    shift = size / 2
    scale = size.max() / 2
    kpts = (kpts - shift) / scale
    return kpts.astype(np.float32)

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)

# def lightglue_registration(img_ref, img_tag, max_size=1280):
#     """
#     lightglue纠偏算法
#     https://github.com/cvg/LightGlue/tree/main
#     https://colab.research.google.com/github/cvg/LightGlue/blob/main/demo.ipynb#scrollTo=6JA4sWG9PV7M
#     M = [[a b tx][c d ty]] a、b、c、d负责旋转、缩放、剪切等线性变换，t_x, t_y是平移量
#     """
#     # 若图片最长边大于max_size，将图片resize到max_size内
#     shape_max = max(list(img_ref.shape[:2]) + list(img_tag.shape[:2]))
#     resize_rate = math.ceil(shape_max / max_size) 
#     H, W = img_tag.shape[:2]
#     img_tag = cv2.resize(img_tag, (int(W / resize_rate), int(H / resize_rate)))
#     H, W = img_ref.shape[:2]
#     img_ref = cv2.resize(img_ref, (int(W / resize_rate), int(H / resize_rate)))
#     resize_scale = [1, 1, resize_rate] # 原始图片相对于resize后的图片的偏移矩阵M的关系。1*3的列表
#     M_scale = np.diag(np.array(resize_scale))  # 把1*3的列表转成3*3的对角矩阵，用于调整仿射变换矩阵M，将其恢复到原始图像尺寸。

#     img_ref = numpy_image_to_torch(img_ref)
#     img_tag = numpy_image_to_torch(img_tag)
#     feats0 = extractor.extract(img_ref.to(device))
#     feats1 = extractor.extract(img_tag.to(device))
#     ref_total_keypoints = feats0["keypoints"].shape[1]
#     tag_total_keypoints = feats1["keypoints"].shape[1]

#     matches01 = matcher({"image0": feats0, "image1": feats1}) #matches01为两张图中特征点的匹配结果

#     feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

#     kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
#     mkpts0, mkpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]] #匹配成功的点的坐标

#     mkpts0 = mkpts0.cpu().numpy()
#     mkpts1 = mkpts1.cpu().numpy()
#     print(f'len(mkpts0):{len(mkpts0)}')
#     if len(mkpts0) <= 10:
#         return None
#     # 此时M只是缩放后的图的映射
#     M, mask = cv2.estimateAffinePartial2D(mkpts0, mkpts1, method=cv2.RANSAC, ransacReprojThreshold=5)

#     if M is None:
#         return None
    
#     M = np.dot(M, M_scale) # 偏移矩阵还原会原始图片对应的M

#     return M
    # if judge_is_matching(len(mkpts0), ref_total_keypoints, tag_total_keypoints):
    #     save_lightglue_img(img_ref, img_tag, mkpts0, mkpts1, save_path = 'lightglue_img.jpg')
    #     return M
    # else:
    #     return None


def judge_is_matching(num, ref_total_keypoints, tag_total_keypoints):
    # 计算匹配点密度
    match_density = (num / min(ref_total_keypoints, tag_total_keypoints))  
    print(f'匹配点的数量num:{num}, ref_total_keypoints:{ref_total_keypoints}, tag_total_keypoints:{tag_total_keypoints}, 匹配密度: {match_density:.8f}')
    base_threshold = 0.2
    # 设定匹配成功的标准
    if match_density < base_threshold:
        logging.warning("匹配点太少，匹配失败！")
        return None
    return True


def save_lightglue_img(img_ref, img_tag, mkpts0, mkpts1, save_path='/data/home/ckx/adjust_camera/lightglue_img.jpg'):
    # 1. 如果输入是 PyTorch Tensor，先转换成 NumPy
    if isinstance(img_ref, torch.Tensor):
        img_ref = img_ref.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
    if isinstance(img_tag, torch.Tensor):
        img_tag = img_tag.permute(1, 2, 0).cpu().numpy()

    # 2. 确保数据范围是 0-255，并转换为 uint8
    img_ref = (img_ref * 255).clip(0, 255).astype(np.uint8) if img_ref.max() <= 1 else img_ref.astype(np.uint8)
    img_tag = (img_tag * 255).clip(0, 255).astype(np.uint8) if img_tag.max() <= 1 else img_tag.astype(np.uint8)

    # 3. 确保图像是 3 通道彩色图
    if len(img_ref.shape) == 2:  # 灰度图转彩色
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR)
    if len(img_tag.shape) == 2:
        img_tag = cv2.cvtColor(img_tag, cv2.COLOR_GRAY2BGR)

    # 4. 拼接两张图，左侧是 img_ref，右侧是 img_tag
    H1, W1 = img_ref.shape[:2]
    H2, W2 = img_tag.shape[:2]
    H_final = max(H1, H2)
    W_final = W1 + W2

    match_img = np.zeros((H_final, W_final, 3), dtype=np.uint8)
    match_img[:H1, :W1] = img_ref
    match_img[:H2, W1:W1+W2] = img_tag  # 右侧是 img_tag

    # 5. 画匹配点和连线
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(mkpts0))]

    for i, (pt1, pt2) in enumerate(zip(mkpts0, mkpts1)):
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]) + W1, int(pt2[1])  # 右侧坐标需要平移

        color = colors[i]  # 确保相邻点颜色不同
        cv2.circle(match_img, (x1, y1), 5, color, -1)  # 在左图画点
        cv2.circle(match_img, (x2, y2), 5, color, -1)  # 在右图画点
        cv2.line(match_img, (x1, y1), (x2, y2), color, 2)  # 画线连接匹配点

    # 6. 保存匹配可视化图像
    cv2.imwrite(save_path, match_img)
    print(f"匹配图已保存到 {save_path}")

    return 0

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

def comput_pt(M, pt, mode):
    ref_coor = (960,540)
    tag_coor = convert_coor(ref_coor, M)
    print(f'ref_coor:{ref_coor}')
    print(f'tag_coor:{tag_coor}')
    x_long = abs(tag_coor[0] - ref_coor[0])/1920
    y_long = abs(tag_coor[1] - ref_coor[1])/1080
    if (x_long < 0.01 and y_long < 0.01) or tag_coor[0] < 0 or tag_coor[1] < 0 :
        return False
    pt_out = (1/x_long)*int(pt[0]) if mode == 'p' else (1/y_long)*int(pt[1])
    return pt_out

def comput_z(M):
    ref_coor = (1200,900)
    tag_coor = convert_coor(ref_coor, M)
    print(f'ref_coor:{ref_coor}')
    print(f'tag_coor:{tag_coor}')
    z = abs((960 - tag_coor[0]) / (960 - ref_coor[0]))
    return z

# 反函数模型
def func1(x, k, b):
    return k / x + b
# 线性函数模型
def func2(x, k, b):
    return k*x + b

def CurveFitting(x_data, y_data, mode):
    if mode == 'pt':
        parameters, _ = curve_fit(func1, x_data, y_data, maxfev=5000)
    elif mode == 'z':
        parameters, _ = curve_fit(func2, x_data, y_data, maxfev=5000)
    return parameters


def get_sub_img(img1, w, h, center_x, center_y, percent=4):
    """
    根据 center_x, center_y 计算出所占比例为 percent 时的图片位置，并将其他区域设为黑色。
    如果某个方向扩展受限，则在相反方向补偿，确保最终区域大小恒定。

    :param img1: 输入图像
    :param w: 图像宽度
    :param h: 图像高度
    :param center_x: 归一化中心点 x 坐标 (0~1)
    :param center_y: 归一化中心点 y 坐标 (0~1)
    :param percent: 目标区域的边长占原图的比例 (percent=4 表示边长为 1/4，实际面积为 1/16)
    :return: 仅保留该区域的图像，其余部分填充为黑色
    """

    # 计算目标区域的宽高（边长 = 1 / percent）
    box_w = w // percent
    box_h = h // percent

    # 计算中心点的像素位置
    x_center = int(center_x * w)
    y_center = int(center_y * h)

    # 计算左右、上下需要扩展的长度
    expand_x = box_w // 2
    expand_y = box_h // 2

    # 计算初步的区域范围
    x_start, x_end = x_center - expand_x, x_center + expand_x
    y_start, y_end = y_center - expand_y, y_center + expand_y

    # 处理 x 方向的边界问题
    if x_start < 0:
        extra_right = abs(x_start)  # 需要补偿的长度
        x_start = 0
        x_end = min(w, x_end + extra_right)  # 向右补偿
    if x_end > w:
        extra_left = x_end - w  # 需要补偿的长度
        x_end = w
        x_start = max(0, x_start - extra_left)  # 向左补偿

    # 处理 y 方向的边界问题
    if y_start < 0:
        extra_down = abs(y_start)  # 需要补偿的长度
        y_start = 0
        y_end = min(h, y_end + extra_down)  # 向下补偿
    if y_end > h:
        extra_up = y_end - h  # 需要补偿的长度
        y_end = h
        y_start = max(0, y_start - extra_up)  # 向上补偿

    # 创建一个全黑图像
    sub_img = np.zeros_like(img1)

    # 仅保留计算出的区域
    sub_img[y_start:y_end, x_start:x_end] = img1[y_start:y_end, x_start:x_end]

    return sub_img


class GetInputData:
    """
    获取巡视输入信息。
    """

    def __init__(self, data):
        self.data = data
        self.center, self.rectangle_coords = self.get_rectangle_info()
        self.p, self.t, self.z = self.get_ptz()
        self.fov_h, self.fov_v, self.use_fov = self.get_fov()
        self.direction_p, self.direction_t = self.get_direction()
        self.range_p, self.range_t, self.range_z = self.get_range()
        self.parameters = self.get_parameters()
        self.img_tag = self.get_img_tag()
        self.img_ref = self.get_img_ref()
        self.img1 = self.get_img1()
        self.max_rate = self.get_max_rate()
        self.requestId = self.get_requestId()
        self.requsetUrl = self.get_requestHostIp_and_requestHostPort()

        # if "img_ref" in self.data and "img_tag" in self.data:
        #     self.save_json()

        
    def get_rectangle_info(self):
        if "rectangle_coords" in self.data and isinstance(self.data["rectangle_coords"], list):
            rec = self.data["rectangle_coords"]
            print(f'rec:{rec}')
            center = [(rec[2] + rec[0]) / 2, (rec[3] + rec[1]) / 2]
            w, h = rec[2] - rec[0], rec[3] - rec[1]

            # 1. 先确保 new_w, new_h 被正确赋值
            new_w, new_h = w, h  

            # 2. 如果矩形太小，放大到最长边 0.05
            if max(w, h) <= 0.05:
                scale = 0.05 / max(w, h)
                new_w, new_h = w * scale, h * scale
                rec = [center[0] - new_w / 2, center[1] - new_h / 2,
                    center[0] + new_w / 2, center[1] + new_h / 2]
                print(f'由于rec太小，已经扩大为:{rec}')

            # 3. 计算是否超出边界
            x_min, y_min, x_max, y_max = rec
            if x_min < 0 or y_min < 0 or x_max > 1 or y_max > 1:
                # 计算缩放比例（确保不超出边界）
                scale_x = 1 / (x_max - x_min) if x_max - x_min > 1 else 1
                scale_y = 1 / (y_max - y_min) if y_max - y_min > 1 else 1
                scale = min(1, scale_x, scale_y)  # 选择最小的缩放比例，最多缩放到1倍

                # 缩放矩形
                new_w *= scale
                new_h *= scale
                rec = [center[0] - new_w / 2, center[1] - new_h / 2,
                    center[0] + new_w / 2, center[1] + new_h / 2]
                
                # 4. 进行最终的边界裁剪，确保不超出
                x_min, y_min, x_max, y_max = rec
                shift_x = max(0, -x_min) - max(0, x_max - 1)  # 计算需要偏移的量
                shift_y = max(0, -y_min) - max(0, y_max - 1)

                rec = [x_min + shift_x, y_min + shift_y, x_max + shift_x, y_max + shift_y]
                
                print(f'矩形超出边界，已调整为: {rec}')

            return center, rec
        else:
            return None, None
        
    def get_ptz(self):
        if "ptz_coords" in self.data and isinstance(self.data["ptz_coords"], list):
            p, t, z = self.data["ptz_coords"]
            return p, t, z
        else:
            return None, None, None
    
    def get_fov(self):
        use_fov = True
        if "Horizontal" in self.data and isinstance(self.data["Horizontal"], (int, float)) and type(self.data["Horizontal"]) is not bool:
            fov_h = self.data["Horizontal"]
        elif "horizontal" in self.data and isinstance(self.data["horizontal"], (int, float)) and type(self.data["Horizontal"]) is not bool:
            fov_h = self.data["horizontal"]
        else:
            use_fov = False
            fov_h = 57
        if "Vertical" in self.data and isinstance(self.data["Vertical"], (int, float)) and type(self.data["Horizontal"]) is not bool:
            fov_v = self.data["Vertical"]
        elif "vertical" in self.data and isinstance(self.data["vertical"], (int, float)) and type(self.data["Horizontal"]) is not bool:
            fov_v = self.data["vertical"]
        else:
            fov_v = 34
        return fov_h, fov_v, use_fov
    
    def get_direction(self):
        if "direction_p" in self.data and (isinstance(self.data["direction_p"], int) or isinstance(self.data["direction_p"], str)):
            direction_p = int(self.data["direction_p"])
        else:
            direction_p = 1
        if "direction_t" in self.data and (isinstance(self.data["direction_t"], int) or isinstance(self.data["direction_t"], str)):
            direction_t = int(self.data["direction_t"])
        else:
            direction_t = 1
        return direction_p, direction_t
    
    def get_range(self):
        if "range_p" in self.data and isinstance(self.data["range_p"], list):
            range_p = self.data["range_p"]
        else:
            range_p = [0, 360]
        if "range_t" in self.data and isinstance(self.data["range_t"], list):
            range_t = self.data["range_t"]
        else:
            range_t = [0, 90]
        if "range_z" in self.data and isinstance(self.data["range_z"], list):
            range_z = self.data["range_z"]
        else:
            range_z = [1, 25]
        
        return range_p, range_t, range_z
    
    def get_parameters(self):
        if "parameters" in self.data and (self.data["parameters"], list):
            parameters = [float(i) for i in self.data["parameters"]]
        else:
            parameters = [60.72370136807651,3.738133238264154,33.575139298346286,1.876545296238718,0.7640445564848101,0.26544800688111997]
        return parameters
    
    def get_img_tag(self):
        if "img_tag" in self.data and isinstance(self.data["img_tag"], str):
            img_t = self.data["img_tag"]
        else:    
            return None
        return get_img(img_t)
    
    def get_img_ref(self):
        if "img_ref" in self.data and isinstance(self.data["img_ref"], str):
            img_r = self.data["img_ref"]
        else:
            return None
        
        return get_img(img_r)
    
    def get_img1(self):
        if "img1" in self.data and isinstance(self.data["img1"], np.ndarray):
            img_r = self.data["img1"]
        elif "img1" in self.data and isinstance(self.data["img1"], str):
            img_r = get_img(self.data["img1"])
        else:
            return None
        return img_r
    
    def get_max_rate(self):
        if "max_rate" in self.data and isinstance(self.data["max_rate"], float):
            img_r = self.data["max_rate"]
        else:
            return 0.8
        
        return img_r
    
    def get_requestId(self):
        if "requestId" in self.data and isinstance(self.data["requestId"], str):
            requestId = self.data["requestId"]
        else:
            return None
        
        return requestId
    
    def get_requestHostIp_and_requestHostPort(self):
        if "requestHostIp" in self.data and isinstance(self.data["requestHostIp"], str) and "requestHostPort" in self.data and \
            isinstance(self.data["requestHostPort"], str):
            requestHostIp = self.data["requestHostIp"]
            requestHostPort = self.data["requestHostPort"]
            url = 'http://' + requestHostIp + ':' + requestHostPort + '/channel/getAlgPtzImg'
        else:
            return None
        return url
    
    def save_json(self):
        img_ref = img2base64(self.get_img_ref())
        img_tag = img2base64(self.get_img_tag())
        ptz_coords = self.data["ptz_coords"]
        parameters = self.data["parameters"]
        direction_p = self.data["direction_p"]
        direction_t = self.data["direction_t"]
        input_data = {'img_ref': img_ref, 'img_tag': img_tag, 'ptz_coords': ptz_coords, 'parameters': parameters, \
                      'direction_p': direction_p, 'direction_t': direction_t}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")  # 格式为 YYYYMMDD_HHMMSS微秒
        file_name = "/data/home/ckx/adjust_camera/data/" + f"input_data_{timestamp}.json"

        with open(file_name, 'w', encoding='utf-8') as json_file:
            json.dump(input_data, json_file, ensure_ascii=False, indent=4)

        print(f"数据已保存为 {file_name}")

def delete_old_json_files(directory, max_age_seconds=3600):
    """
    删除指定目录下超过 max_age_seconds 的 JSON 文件。

    :param directory: 目录路径
    :param max_age_seconds: 最大存活时间，单位为秒（默认1分钟）
    """
    current_time = time.time()  # 当前时间（秒）
    
    # 确保目录存在
    if not os.path.exists(directory):
        print(f"目录 {directory} 不存在！")
        return

    # 遍历目录中的文件
    for filename in os.listdir(directory):
        # 检查是否是 JSON 文件
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            
            # 获取文件的修改时间
            file_mod_time = os.path.getmtime(file_path)
            
            # 计算文件年龄
            file_age = current_time - file_mod_time

            # 如果文件超过 max_age_seconds，则删除
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    print(f"已删除文件: {file_path}")
                except Exception as e:
                    print(f"删除文件 {file_path} 时出错: {e}")


if __name__ == '__main__':
    print('test')