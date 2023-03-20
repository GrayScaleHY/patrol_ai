import os
import time
import cv2
import json
from lib_image_ops import base642img, img2base64, img_chinese
from lib_sift_match import sift_create, sift_match, correct_offset, detect_diff
import base64
import hashlib

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def img2base64_(img_file):
    """
    numpy的int数据转换为base64格式。
    """
    f = open(img_file, "rb")
    lines = f.read()
    f.close()
    img_base64 = base64.b64encode(lines)
    img_base64 = img_base64.decode()
    return img_base64

def checkout_md5(base64_tag, base64_ref):
    """
    查看输入的图片组是否在md5字典中
    """
    if not os.path.exists("md5_dict_big.json"):
        return None, "No_md5_dict"
    f = open("md5_dict_big.json", "r", encoding='utf-8')
    md5_dict = json.load(f)
    f.close()

    md5_tag = hashlib.md5(base64.b64decode(base64_tag)).hexdigest()
    md5_ref = hashlib.md5(base64.b64decode(base64_ref)).hexdigest()
    md5_match1 = md5_tag + " : " + md5_ref
    md5_match2 = md5_ref + " : " + md5_tag
    if md5_match2 in md5_dict:
        tag_diff = md5_dict[md5_match2]["box"]
        name = md5_dict[md5_match2]["name"]
    elif md5_match1 in md5_dict:
        tag_diff = md5_dict[md5_match1]["box"]
        name = md5_dict[md5_match1]["name"]
    elif md5_tag == md5_ref:
        tag_diff = []; name = "SAME_MAD5"
    else:
        tag_diff = None; name = "NO_MATCH"
    return tag_diff, name

def panbie_main(img_ref, img_tag):
    """
    使用判别算法求差异区域
    """
    ## resize, 降低分别率，加快特征提取的速度。
    resize_rate = 2
    H, W = img_ref.shape[:2]  ## resize
    img_ref = cv2.resize(img_ref, (int(W / resize_rate), int(H / resize_rate)))
    feat_ref = sift_create(img_ref) # 提取sift特征

    H, W = img_tag.shape[:2]  ## resize
    img_tag = cv2.resize(img_tag, (int(W / resize_rate), int(H / resize_rate)))
    feat_tag = sift_create(img_tag) # 提取sift特征

    M = sift_match(feat_tag, feat_ref, ratio=0.5, ops="Affine")
    img_ref, cut = correct_offset(img_ref, M, b=True)
    img_tag = img_tag[cut[1]:cut[3], cut[0]:cut[2], :]
    img_ref = img_ref[cut[1]:cut[3], cut[0]:cut[2], :]
    diff_area = detect_diff(img_ref, img_tag)
    if len(diff_area) != 0:  
        diff_area = [diff_area[0] + cut[0], diff_area[1] + cut[1], diff_area[2] + cut[0], diff_area[3] + cut[1]]
    # tag_diff = identify_defect(img_ref, feat_ref, img_tag, feat_tag) # 判别算法
    diff_area = [int(d * resize_rate) for d in diff_area] ## 将tag_diff还原回原始大小
    return diff_area

def get_input_data(input_data):
    """
    提取input_data中的信息。
    return:
        img_tag: 目标图片数据
        img_ref: 模板图片数据
    """

    img_tag = base642img(input_data["image"])

    ## 是否有模板图
    img_ref = None
    if "img_ref" in input_data["config"]:
        if isinstance(input_data["config"]["img_ref"], str):
            img_ref = base642img(input_data["config"]["img_ref"]) 

    return img_tag, img_ref

def inspection_identify_defect(input_data):
    """
    yolov5的目标检测推理。
    """
    ## 将输入请求信息可视化
    TIME_START = time.strftime("%m%d%H%M%S") + "_"
    if "checkpoint" in input_data and isinstance(input_data["checkpoint"], str) and len(input_data["checkpoint"]) > 0:
        TIME_START = TIME_START + input_data["checkpoint"] + "_"
    save_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_path = os.path.join(save_path, "result_patrol", input_data["type"])
    os.makedirs(save_path, exist_ok=True)
    f = open(os.path.join(save_path, TIME_START + "input_data.json"), "w")
    json.dump(input_data, f, ensure_ascii=False)  # 保存输入信息json文件
    f.close()

    ## 初始化输入输出信息。
    img_tag, img_ref = get_input_data(input_data)
    out_data = {"code": 1, "data":[], "img_result": input_data["image"], "msg": "Success request object detect; "} # 初始化out_data
    img_tag_ = img_tag.copy()
    cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag.jpg"), img_tag) # 将输入图片可视化
    cv2.imwrite(os.path.join(save_path, TIME_START + "img_ref.jpg"), img_ref) # 将输入图片可视化

    if img_ref is None:
        out_data["msg"] = out_data["msg"] + "; img_ref not exist;"
        out_data["code"] = 1
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=60)
        out_data["img_result"] = img2base64(img_tag_)
        cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag_cfg.jpg"), img_tag_)
        return out_data

    ## 检查图片是否能用md5匹配
    base64_tag = input_data["image"]
    base64_ref = input_data["config"]["img_ref"]
    tag_diff, name = checkout_md5(base64_tag, base64_ref)

    if tag_diff is None:  
        tag_diff = panbie_main(img_ref, img_tag) ## 使用算法
    else:
        f = open(os.path.join(save_path, TIME_START + name), "w")
        f.close()

    out_cfg = []
    if len(tag_diff) == 0:
        img_tag_ = img_chinese(img_tag_, "正常", (20,10), (0, 255, 0), size=20)
        out_cfg.append({"label": "0", "bbox":[]})
        out_data["code"] = 0
    else:
        rec = tag_diff
        cv2.rectangle(img_tag_, (int(rec[0]), int(rec[1])),(int(rec[2]), int(rec[3])), (0,0,255), thickness=2)
        img_tag_ = img_chinese(img_tag_, "异常", (int(rec[0])+10, int(rec[1])+20), (0,0,255), size=20)
        out_cfg.append({"label": "1", "bbox":rec})
    
    out_data["data"] = out_cfg

    ## 可视化计算结果
    f = open(os.path.join(save_path, TIME_START + "out_data.json"), "w")
    json.dump(out_data, f, ensure_ascii=False, indent=2)  # 保存输入信息json文件
    f.close()
    
    ## 输出可视化结果的图片。
    img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=60)
    cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag_cfg.jpg"), img_tag_)
    out_data["img_result"] = img2base64(img_tag_)

    return out_data

if __name__ == '__main__':
    ref_file = "/home/yh/image/python_codes/test/panbie/0002_normal.jpg"
    tag_file = "/home/yh/image/python_codes/test/panbie/0002_1.jpg"

    # img_tag = img2base64(cv2.imread(tag_file))
    # img_ref = img2base64(cv2.imread(ref_file))
    img_tag = img2base64_(tag_file)
    img_ref = img2base64_(ref_file)

    input_data = {"image": img_tag, "config":{"img_ref": img_ref}, "type": "identify_defect"}

    start = time.time()
    out_data = inspection_identify_defect(input_data)
    print(time.time() - start)
    for c_ in out_data:
        if c_ != "img_result":
            print(c_,":",out_data[c_])

