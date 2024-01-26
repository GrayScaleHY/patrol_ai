"""
孪生网络模型推理脚本
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
from BIT_CD.models.basic_model import CDEvaluator
 
from types import SimpleNamespace

def load_bit_model(model_file, device="0"):
    """
    加载BIT-CD模型
    args:
        model_file: 模型路径
        device: 是否使用显卡, egs: "0"、 "0,1,2,3"、 "cpu"
    """
    # parser = ArgumentParser()
    # args_new = parser.parse_args()
    args_new = SimpleNamespace()
    args_new.n_class = 2
    if device == "cpu":
        args_new.gpu_ids = []
    else:
        args_new.gpu_ids = [int(i) for i in device.split(",")]
    args_new.checkpoint_dir = os.path.dirname(model_file)
    args_new.checkpoint_name = os.path.basename(model_file)
    args_new.output_folder = args_new.checkpoint_dir
    args_new.net_G = "base_transformer_pos_s4_dd8"
    model = CDEvaluator(args_new)
    model.load_checkpoint(args_new.checkpoint_name)
    model.eval()

    return model

def inference_bit(model, img_a, img_b, resize=256):
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

    batch = {"A": img_a, "B": img_b}

    ## 推理
    img_diff = model._forward_pass(batch)
    img_diff = img_diff.cpu().numpy()[0][0].astype(np.uint8)
    
    return img_diff

if __name__ == '__main__':
    model_file = "/data/PatrolAi/bit_cd/bit_cd.pt"
    img_tag = "/data/PatrolAi/result_patrol/0002_1.jpg"
    img_ref = "/data/PatrolAi/result_patrol/0002_normal.jpg"
    out_file = "/data/PatrolAi/result_patrol/panbie_resu.png"
    import time

    model = load_bit_model(model_file, device="0")
    img_a = cv2.imread(img_tag)
    img_b = cv2.imread(img_ref)
    
    
    start = time.time()
    img_diff = inference_bit(model, img_a, img_b, resize=256)
    print("spend time:", time.time() - start)

    cv2.imwrite(out_file, img_diff)
