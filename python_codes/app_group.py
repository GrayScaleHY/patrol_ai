
# -*- coding: utf-8 -*-

'''
递归处理复杂结构目录下的图片去重 基于ResNet神经网络
节省内存，加速计算
'''

import os
import time
import numpy as np
from PIL import Image
import torch
import faiss
import glob
from torchvision import models, transforms
from lib_help_base import GetInputData
import requests


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class FeatureExtractor:
    def __init__(self):
        torch.hub.set_dir("~/.cache/torch/hub")
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Identity()
        self.model = self.model.to(device).eval()

    def __call__(self, tensors):
        with torch.no_grad():
            return self.model(tensors.to(device)).cpu()

extractor = FeatureExtractor()

def batch_generator_list(img_list, batch_size):
    """生成图片路径的批次"""
    for i in range(0, len(img_list), batch_size):
        yield img_list[i:i + batch_size]


def process_batch(batch_paths, extractor):
    """处理单个批次返回特征和成功路径"""
    batch_tensors = []
    valid_paths = []

    for path in batch_paths:
        try:
            with Image.open(path) as img:
                batch_tensors.append(transform(img.convert("RGB")))
                valid_paths.append(path)
        except Exception as e:
            print(f"Error loading {path}: {str(e)}")

    if not batch_tensors:
        return None, []

    features = extractor(torch.stack(batch_tensors))
    return features.numpy(), valid_paths

def build_index(features):
    """构建FAISS索引并标准化特征"""
    # 标准化特征
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    features_normalized = features / norms

    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features_normalized.astype(np.float32))
    return features_normalized, index

def find_duplicates(features_normalized, index, threshold=0.9):
    """使用FAISS进行范围搜索查找重复项"""
    print("Start range search...")
    lims, D, I = index.range_search(features_normalized.astype(np.float32), threshold)

    # 使用并查集管理分组
    parent = list(range(len(features_normalized)))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # 路径压缩
            u = parent[u]
        return u

    for i in range(len(features_normalized)):
        if lims[i + 1] - lims[i] > 0:
            for j in I[lims[i]:lims[i + 1]]:
                if j > i:  # 避免重复处理
                    root_i = find(i)
                    root_j = find(j)
                    if root_i != root_j:
                        parent[root_j] = root_i

    # 收集分组
    groups = {}
    for idx in range(len(features_normalized)):
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    return [g for g in groups.values()]
    # return [g for g in groups.values() if len(g) > 1]

def img_group(img_list, batch_size=10, threshold=0.9):
    """
    args:
        img_list: 图片路径列表
        batch_size: 批处理
        threshold: 阈值，值越大，分组越精细
    return:
        group: 分组列表，[["path1", "path2"], ["path3", "path4"], ..]
    """
    all_features = []
    all_paths = []
    total_processed = 0
    batch_paths_list = batch_generator_list(img_list, batch_size)
    for batch_paths in batch_paths_list:
        batch_start = time.time()
        features, valid_paths = process_batch(batch_paths, extractor)

        if features is not None:
            all_features.append(features)
            all_paths.extend(valid_paths)
            total_processed += len(valid_paths)
            print(
                f"Processed {total_processed} images | Last batch: {len(valid_paths)} images ({time.time() - batch_start:.2f}s)")

    if not all_paths:
        print("No valid images found!")
        return
    
    # 合并特征并构建索引
    features = np.concatenate(all_features, axis=0)
    features_normalized, index = build_index(features)
    del all_features, features  # 释放内存

    # 查找重复组
    duplicate_groups = find_duplicates(features_normalized, index, threshold)
    print(f"Found {len(duplicate_groups)} duplicate groups")

    duplicate_groups = sorted(duplicate_groups, key=len, reverse=True)

    groups = []
    for group in duplicate_groups:
        # 获取排序后的完整路径
        sorted_paths = sorted([all_paths[idx] for idx in group])
        groups.append(sorted_paths)

    return groups

def get_img_paths(input_data):
    img_paths = []
    for tag_ in input_data["images_path"]:
        if os.path.exists(tag_):
            img_paths.append(tag_)
        elif tag_.startswith("http"):
            tag_tmp = "/export/" + "/".join(tag_.split("/")[-3:])
            if not os.path.exists(tag_tmp):
                tag_dir = os.path.dirname(tag_tmp)
                os.makedirs(tag_dir, exist_ok=True)
                print("request download--------------------------------------")
                print(tag_)
                print(tag_tmp)
                os.makedirs(tag_dir, exist_ok=True)
                r = requests.get(tag_)
                f = open(tag_tmp, "wb")
                f.write(r.content)
                f.close()
                # wget.download(tag_, tag_tmp)
            img_paths.append(tag_tmp)
        else:
            print("Warning: con not get", tag_ , "!")
            continue
    return img_paths

def patrol_group(input_data):
    DATA = GetInputData(input_data)
    fineness = DATA.fineness
    thr_map = {0: 0.81, 1: 0.85, 2: 0.87, 3: 0.88, 4: 0.89, 5: 0.9, 
               6: 0.91, 7: 0.92, 8: 0.93, 9: 0.95, 10: 0.99}
    img_list = get_img_paths(input_data)

    groups = img_group(img_list, batch_size=50, threshold=thr_map[fineness])

    for i, group in enumerate(groups):
        for j, tmp_path in enumerate(group):
            end_path = tmp_path[8:]
            for real_path in input_data["images_path"]:
                if end_path in real_path:
                    groups[i][j] = real_path
                    break
    
    out_data = {"code": 0, "data": groups, "msg": "Success!"} 
    return out_data              


if __name__ == "__main__":
    img_dir = "images_old"
    img_list = glob.glob(os.path.join(img_dir, "*"))

    input_data = {"images_path": img_list, "config": {"fineness": 5}, "type": "image_group"}
    out_data = patrol_group(input_data)
    print(out_data)

