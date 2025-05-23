import cv2
import numpy as np
from typing import List, Optional, Union

try:
    '''
    加载om模型，注意，onnx模型转om模型命令如下，需要ascend v7.0.1版本
    superpoint.om模型转换：atc --framework=5 --model=superpoint.onnx --input_shape="image:1,1,512,512" --output=superpoint --soc_version=Ascend310P3 --precision_mode=allow_mix_precision --log=error
    lightglue.om 模型转换：atc --framework=5 --model=superpoint_lightglue.onnx --input_shape="kpts0:1,1~2048,2;desc0:1,1~2048,256;kpts1:1,1~2048,2;desc1:1,1~2048,256" --input_format=ND --output=superpoint_lightglue --soc_version=Ascend310P3 --precision_mode=allow_mix_precision --log=error
    '''
    from ais_bench.infer.interface import InferSession 
    extractor_path="/data/PatrolAi/superglue_atlas/superpoint.om"
    lightglue_path="/data/PatrolAi/superglue_atlas/superpoint_lightglue.om"
    extractor = InferSession(0, extractor_path) # 需要ascend v7.0.1版本
    lightglue = InferSession(0, lightglue_path)
except:
    '''
    加载onnx模型，注意pt模型转onnx模型命令如下。
    https://github.com/fabio-sim/LightGlue-ONNX
    python export.py --img_size 512 --extractor_type superpoint --extractor_path weights/superpoint.onnx --lightglue_path weights/superpoint_lightglue.onnx --dynamic
    '''
    import onnxruntime as ort
    extractor_path="/data/PatrolAi/superglue_atlas/superpoint.onnx"
    lightglue_path="/data/PatrolAi/superglue_atlas/superpoint_lightglue.onnx"
    providers=["CPUExecutionProvider", "CPUExecutionProvider"]
    extractor = ort.InferenceSession(extractor_path, providers=providers)
    sess_options = ort.SessionOptions()
    lightglue = ort.InferenceSession(lightglue_path, sess_options=sess_options, providers=providers)

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
    img: np.ndarray,
    grayscale: bool = False,
    resize: int = None,
    fn: str = "max",
    interp: str = "area",
):
    # img = read_image(path, grayscale=grayscale)
    if img.ndim == 3:
        img = img[..., ::-1]
        
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

def lightglue_registration_onnx(img_ref, img_tag, max_size=512, max_pointer=512):
    """
    lightglue onnx推理代码
    https://github.com/fabio-sim/LightGlue-ONNX
    """
    image0, scales0 = load_image(img_ref, resize=max_size)
    image1, scales1 = load_image(img_tag, resize=max_size)
    image0 = rgb_to_grayscale(image0)  # only needed for SuperPoint
    image1 = rgb_to_grayscale(image1)  # only needed for SuperPoint

    kpts0, scores0, desc0 = extractor.run(None, {"image": image0})
    kpts1, scores1, desc1 = extractor.run(None, {"image": image1})

    # 对特征根据score从大到小排序
    idx0 = np.argsort(-scores0, axis=1)[0]
    kpts0 = kpts0[:,idx0,:]; scores0 = scores0[:,idx0]; desc0 = desc0[:,idx0,:]
    idx1 = np.argsort(-scores1, axis=1)[0]
    kpts1 = kpts1[:,idx1,:]; scores1 = scores1[:,idx1]; desc1 = desc1[:,idx1,:]

    max_p = max_pointer
    if max_p:
        kpts0 = kpts0[:,:max_p,:]; scores0 = scores0[:,:max_p]; desc0 = desc0[:,:max_p,:]
        kpts1 = kpts1[:,:max_p,:]; scores1 = scores1[:,:max_p]; desc1 = desc1[:,:max_p,:]
        kpts0_nm = normalize_keypoints(kpts0, image0.shape[2], image0.shape[3])
        kpts1_nm = normalize_keypoints(kpts1, image1.shape[2], image1.shape[3])

    kpts0_nm = normalize_keypoints(kpts0, image0.shape[2], image0.shape[3])
    kpts1_nm = normalize_keypoints(kpts1, image1.shape[2], image1.shape[3])

    matches0, mscores0 = lightglue.run(None,{"kpts0": kpts0_nm, "kpts1": kpts1_nm, "desc0": desc0,"desc1": desc1})

    if len(matches0) < 3:
        return None

    m_kpts0, m_kpts1 = post_process(kpts0, kpts1, matches0, scales0, scales1)
    
    M, mask = cv2.estimateAffinePartial2D(m_kpts0, m_kpts1, method=cv2.RANSAC, ransacReprojThreshold=5)

    return M

def lightglue_registration_om(img_ref, img_tag, max_size=512, max_pointer=512):
    """
    lightglue onnx推理代码
    https://github.com/fabio-sim/LightGlue-ONNX
    """
    image0, scales0 = load_image(img_ref, resize=max_size)
    image1, scales1 = load_image(img_tag, resize=max_size)
    image0 = rgb_to_grayscale(image0)  # only needed for SuperPoint
    image1 = rgb_to_grayscale(image1)  # only needed for SuperPoint

    kpts0, scores0, desc0 = extractor.infer([image0], mode="dymshape", custom_sizes=[2000000, 2000000, 2000000])
    kpts1, scores1, desc1 = extractor.infer([image1], mode="dymshape", custom_sizes=[2000000, 2000000, 2000000])

    # 对特征根据score从大到小排序
    idx0 = np.argsort(-scores0, axis=1)[0]
    kpts0 = kpts0[:,idx0,:]; scores0 = scores0[:,idx0]; desc0 = desc0[:,idx0,:]
    idx1 = np.argsort(-scores1, axis=1)[0]
    kpts1 = kpts1[:,idx1,:]; scores1 = scores1[:,idx1]; desc1 = desc1[:,idx1,:]

    max_p = max_pointer
    if max_p:
        kpts0 = kpts0[:,:max_p,:]; scores0 = scores0[:,:max_p]; desc0 = desc0[:,:max_p,:]
        kpts1 = kpts1[:,:max_p,:]; scores1 = scores1[:,:max_p]; desc1 = desc1[:,:max_p,:]
        kpts0_nm = normalize_keypoints(kpts0, image0.shape[2], image0.shape[3])
        kpts1_nm = normalize_keypoints(kpts1, image1.shape[2], image1.shape[3])

    matches0,  mscores0 = lightglue.infer([kpts0_nm, kpts1_nm, desc0, desc1], mode='dymshape', custom_sizes=[200000, 200000])

    if len(matches0) < 3:
        return None

    m_kpts0, m_kpts1 = post_process(kpts0, kpts1, matches0, scales0, scales1)
    
    M, mask = cv2.estimateAffinePartial2D(m_kpts0, m_kpts1, method=cv2.RANSAC, ransacReprojThreshold=5)

    return M

if __name__ == '__main__':
    import time

    ref_file = "ref.jpg"
    tag_file = "tag.jpg"
    
    img_tag= cv2.imread(tag_file)
    img_ref = cv2.imread(ref_file)

    H, W = img_tag.shape[:2]
    img_tag = cv2.resize(img_ref, (int(W / 2), int(H / 2)))

    start = time.time()
    M = lightglue_registration_om(img_ref, img_tag, max_size=512)
    print("spend time:", time.time() - start)

    print(M)

    seg_raw = [100, 400, 150, 450]
    seg = [100, 400, 150, 450]
    cv2.line(img_ref, (int(seg[0]), int(seg[1])),(int(seg[2]), int(seg[3])), (0, 255, 0), 2)
    seg[:2] = convert_coor(seg[:2], M)
    seg[2:4] = convert_coor(seg[2:4], M)
    cv2.line(img_tag, (int(seg[0]), int(seg[1])),(int(seg[2]), int(seg[3])), (0, 255, 0), 2)
    cv2.imwrite("img_tag.jpg", img_tag)
    cv2.imwrite("img_ref.jpg", img_ref)

    print(1)
