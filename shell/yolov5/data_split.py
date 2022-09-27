"""
将数据集按90%train和10%val的比例分配。并验证label文件是否有内容以及是否有对应的.jpg文件
使用步骤：
    1. 最终输出文件夹<data_dir>中新建一个new_data目录，并且将图片和标签放入new_data目录下
    2. 执行脚本 python data_split.py --dir <data_dir>
"""

import os
import glob
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dir',
    type=str,
    default='./',
    help='data path ')
args, unparsed = parser.parse_known_args()

data_path = args.dir #数据根目录

start = time.time()
## 将所有图片转移到data目录下
new_data_dir = os.path.join(data_path, "new_data")
assert os.path.exists(new_data_dir), new_data_dir + " is not exist !"
for type_ in ["images", "labels"]:
    for root, dirs, files in os.walk(os.path.join(data_path, type_)):
        for file_name in files:
            in_file = os.path.join(root, file_name)
            out_file = os.path.join(new_data_dir, file_name)
            if os.path.exists(out_file):
                os.remove(in_file)
            else:
                os.rename(in_file, out_file)

## 验证new_data目录下的文件并分配到images和labels中
os.makedirs(os.path.join(data_path, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(data_path, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(data_path, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(data_path, "labels", "val"), exist_ok=True)
count = 0
for txt_file in glob.glob(os.path.join(new_data_dir, "*.txt")):

    if count % 10 == 0:
        type_ = "val"
    else:
        type_ = "train"

    img_file = txt_file[:-4] + ".jpg"
    txt_name = os.path.basename(txt_file)
    img_name = os.path.basename(img_file)

    ## 若不存在对应的图片，将label文件移到最外层
    if not os.path.exists(img_file):
        txt_out = os.path.join(data_path,txt_name)
        os.rename(txt_file, txt_out)
        continue

    f = open(txt_file, "r", encoding='utf-8')
    txt_con = f.read()
    f.close()

    ## 若label文件内容太少，将label文件和图片文件移到最外层
    if len(txt_con) > 3:
        txt_out = os.path.join(data_path, "labels", type_, txt_name)
        img_out = os.path.join(data_path, "images", type_, img_name)
        count += 1
    else:
        txt_out = os.path.join(data_path, txt_name)
        img_out = os.path.join(data_path, img_name)
    
    os.rename(txt_file, txt_out)
    os.rename(img_file, img_out)

print("Split data spend time:", time.time() - start)


