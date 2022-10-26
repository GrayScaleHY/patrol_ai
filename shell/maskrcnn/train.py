from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
import os
import sys
sys.path.append('./')
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import os

cfg_file = "/data/maskrcnn/pointer/pointer_R_101_FPN.yaml"
train_set = "/data/maskrcnn/pointer/train"
train_set_name = "meter"
label_list = ["pointer"]
gpu = 0
num_workers = 8
steps_lr = (70000,100000)
max_iter = 120000
batch_size = 35
save_dir = "/data/maskrcnn/pointer/saved_model/2022-06-20"
os.makedirs(save_dir, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

## 加载数据集
label_file = os.path.join(train_set, "annotations/trainval.json")
img_dir = os.path.join(train_set, "images")
register_coco_instances(train_set_name, {}, label_file, img_dir)
meter_metadata = MetadataCatalog.get(train_set_name)
dataset_dicts = DatasetCatalog.get(train_set_name)
meter_metadata.thing_classes = label_list
# print(dataset_dicts)

## 加载config文件
cfg = get_cfg()
cfg.merge_from_file(cfg_file)
cfg.MODEL.WEIGHTS = '/data/maskrcnn/R-101.pkl'
cfg.DATASETS.TRAIN = (train_set_name,) # train sets
cfg.DATASETS.TEST = ()   # val sets
cfg.DATALOADER.NUM_WORKERS=num_workers
cfg.SOLVER.STEPS = steps_lr
cfg.SOLVER.MAX_ITER = max_iter    # 300 iterations seems good enough, but you can certainly train longer
cfg.SOLVER.IMS_PER_BATCH = batch_size  # batch size
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(label_list)  # 类别数目
cfg.OUTPUT_DIR = save_dir

## 训练模型
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()