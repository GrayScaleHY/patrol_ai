"""
孪生网络模型推理脚本
pt模型转om模型说明: https://git.utapp.cn/yuanhui/BIT_CD/-/blob/bit_huawei/README.md
"""

import numpy as np
import torch
import os
import cv2
# import sys
# sys.path.insert(0,'BIT_CD') ## ultralytics/yolov5 存放的路径
# from datasets.data_utils import CDDataAugmentation
# from models.basic_model import CDEvaluator
from BIT_CD.datasets.data_utils import CDDataAugmentation
# from BIT_CD.models.basic_model import CDEvaluator
from ais_bench.infer.interface import InferSession

def load_bit_model(model_file, device="0"):
    """
    加载BIT-CD模型
    args:
        model_file: 模型路径
        device: 是否使用显卡, egs: 0
    """
    if model_file.endswith(".pt"):
        model_file = model_file[:-3] + ".om"
    model = InferSession(int(device), model_file)
    return model

def inference_bit(model, img_a, img_b, resize=512):
    """
    BIT_CD模型推理
    args:
        model: 通过load_bit_model加载的模型
        img_a: cv2读取的A图片
        img_b: cv2读取的B图片
        resize: 送进网络的图片大小
    """
    H, W = img_a.shape[:2]
    ## 数据前处理
    img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
    img_l = np.zeros([resize, resize], dtype=np.uint8)
    img_l = img_l // 255

    ## 图片转为tensor并resize
    augm = CDDataAugmentation(resize)
    [img_a, img_b], [img_l] = augm.transform([img_a, img_b], [img_l], to_tensor=True)
    img_a = torch.unsqueeze(img_a, dim=0) 
    img_b = torch.unsqueeze(img_b, dim=0) 
    
    img_a = torch.unsqueeze(img_a, dim=0) 
    img_b = torch.unsqueeze(img_b, dim=0)

    batch = torch.cat([img_a, img_b])
    
    ## 推理
    input_img = batch.numpy().astype(np.float32)
    img_diff = model.infer([input_img])
    
    img_diff = torch.tensor(img_diff[0])
    img_diff = torch.argmax(img_diff, dim=1, keepdim=True)
    img_diff = img_diff * 255
    img_diff = img_diff.numpy()

    img_diff = img_diff[0][0].astype(np.uint8)
    
    return img_diff

if __name__ == '__main__':
    import time
    model_file = "/data/PatrolAi/bit_cd/bit_cd.om"
    model = load_bit_model(model_file, device=0)

    img_tag = "/data/PatrolAi/result_saved/0002_1.jpg"
    img_ref = "/data/PatrolAi/result_saved/0002_normal.jpg"
    out_file = "/data/PatrolAi/result_saved/panbie_result1.jpg"  
    
    img_a = cv2.imread(img_tag)
    img_b = cv2.imread(img_ref)
    start = time.time()
    img_diff = inference_bit(model, img_a, img_b, resize=512)
    print("spend time:", time.time() - start)

    cv2.imwrite(out_file, img_diff)
