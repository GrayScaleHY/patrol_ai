
from lib_image_ops import base642img, img2base64, img_chinese
from util_yjsk import disconnector_state, json2bboxes
import glob
import time
import json
import cv2
import numpy as np
import os
from lib_sift_match import sift_create

if os.path.exists("feats.npy"):
    FEATS = np.load("feats.npy", allow_pickle=True).item()
else:
    FEATS = {}

def create_npy(dir_):
    """
    创建sift特征列表和文件名列表
    args:
        dir: 待提取特征的文件夹路径
    """
    feats = {}
    for img_file in glob.glob(os.path.join(dir_, "*.png")):
        print(img_file)
        img_name = os.path.basename(img_file)
        img = cv2.imread(img_file)
        H, W = img.shape[:2]
        img = cv2.resize(img, (int(W / 4), int(H / 4)))
        _, feat = sift_create(img)
        feats[img_name] = feat
    np.save("feats.npy", feats)

def sift_match_good(feat1, feat2, ratio=0.5, ops="Affine"):
    """
    使用sift特征，flann算法计算两张轻微变换的图片的的偏移转换矩阵M。
    args:
        feat1, feat2: 两图片的sift特征
        ratio: sift点正匹配的阈值
        ops: 变换的方式，可选择"Affine"(仿射), "Perspective"(投影)
    return:
        good_n: good match的数量
    """

    if  feat1 is None or feat2 is None or len(feat1) == 0 or len(feat2) == 0:
        print("warning: img have no sift feat!")
        return 0
    ## 画出siftt特征点
    # ref_sift = cv2.drawKeypoints(ref_img,kps1,ref_img,color=(255,0,255)) # 画sift点
    # tar_sift = cv2.drawKeypoints(tag_img,kps2,tag_img,color=(255,0,255))
    # hmerge = np.hstack((ref_sift, tar_sift)) # 两张图拼接在一起
    # cv2.imwrite("images/test_sift.jpg", hmerge)
    
    ## flann 快速最近邻搜索算法，计算两张特征的正确匹配点。
    ## https://www.cnblogs.com/shuimuqingyang/p/14789534.html
    ## 使用gpu计算sift matches
    print("Warning: sift match with cpu !!")
    flann_index_katree = 1
    index_params = dict(algorithm=flann_index_katree, trees=5) # trees:指定待处理核密度树的数量
    search_params = dict(checks=50) # checks: 指定递归遍历迭代的次数
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(feat1, feat2, k=2)

    # 画出匹配图
    # img_match = cv2.drawMatchesKnn(ref_img,kps1,tag_img,kps2,matches,None,flags=2)
    # cv2.imwrite("images/test_match.jpg",img_match)

    ## store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    # print("num of good match pointer:", len(good))
    return len(good)

state_map = {
    "合-合": {"name": "合闸正常", "color": [(255,0,0), (255,0,0)]},
    "分-合": {"name": "合闸正常", "color": [(255,0,0), (255,0,0)]},
    "异常-合": {"name": "合闸正常", "color": [(255,0,0), (255,0,0)]},
    "无法判别状态-合": {"name": "合闸正常", "color": [(255,0,0), (255,0,0)]},
    "分-分": {"name": "分闸正常", "color": [(0,255,0), (0,255,0)]},
    "合-分": {"name": "分闸正常", "color": [(0,255,0), (0,255,0)]},
    "异常-分": {"name": "分闸正常", "color": [(0,255,0), (0,255,0)]},
    "无法判别状态-分": {"name": "分闸正常", "color": [(0,255,0), (0,255,0)]},
    "合-异常": {"name": "分闸异常", "color": [(255,0,255), (255,0,255)]},
    "分-异常": {"name": "合闸异常", "color": [(0,0,255), (0,0,255)]},
    "异常-异常": {"name": "合闸异常", "color": [(0,0,255), (0,0,255)]},
    "无法判别状态-异常": {"name": "合闸异常", "color": [(0,0,255), (0,0,255)]},
    "无法判别状态-无法判别状态": {"name": "合闸异常", "color": [(0,0,255), (0,0,255)]}
}

def find_models(img_tag_end):
    H, W = img_tag_end.shape[:2]
    img_tag_end = cv2.resize(img_tag_end, (int(W / 4), int(H / 4)))
    _, feat = sift_create(img_tag_end)
    good_max = 0
    img_name = ""
    for _name in FEATS:
        
        feat_ref = FEATS[_name]
        good_num = sift_match_good(feat, feat_ref)
        if good_num > good_max:
            good_max = good_num
            img_name = _name
    return img_name
    
def inspection_disconnector_video(input_data):
    """
    刀闸识别
    """
    cfg_dir = "/export/patrolservice/VIDEO"

    TIME_START = time.strftime("%m-%d-%H-%M-%S") + "_"
    save_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_path = os.path.join(save_path, "result_patrol", input_data["type"])
    os.makedirs(save_path, exist_ok=True)
    f = open(os.path.join(save_path, TIME_START + "input_data.json"), "w")
    json.dump(input_data, f, ensure_ascii=False)  # 保存输入信息json文件
    f.close()
    
    ## 提取data信息
    out_data = {"code": 0, "data":{}, "msg": "Success request disconnector"}
    video_path = input_data["video_path"]

    ## 截取视频第一帧和最后一帧
    cap = cv2.VideoCapture(video_path) ## 建立视频对象
    count = 0 
    while(cap.isOpened()):
        ret, frame = cap.read() # 逐帧读取
        if frame is not None:
            if count < 1:
                img_tag_start = frame
            else:
                img_tag_end = frame
            count += 1
        if not ret:
            break
    
    ## 求刀闸状态
    json_list = glob.glob(os.path.join(cfg_dir, "*.json"))
    if len(json_list) < 1:
        start_ = time.time()
        img_name = find_models(img_tag_end)
        print("sift match spend time:", time.time() - start_)
        print("match name:", img_name)
        json_file = os.path.join("cfgs",img_name[:-7] + ".json")
        open_file = os.path.join("cfgs", img_name[:-7] + "_off.png")
        close_file = os.path.join("cfgs", img_name[:-7] + "_on.png")
    else:
        json_file = glob.glob(os.path.join(cfg_dir, "*.json"))[0]
        open_file = glob.glob(os.path.join(cfg_dir, "*_off.png"))[0]
        close_file = glob.glob(os.path.join(cfg_dir, "*_on.png"))[0]
        assert os.path.exists(json_file) and os.path.exists(open_file) and os.path.exists(close_file), "模板文件不全"
    img_open = cv2.imread(open_file)
    img_close = cv2.imread(close_file)
    img_opens = [img_open]
    img_closes = [img_close]
    bboxes = json2bboxes(json_file, img_open)[:2]

    ## 分析头尾两帧的分合状态
    state_start, _, bboxes_tag_start = disconnector_state(img_tag_start, img_opens, img_closes, box_state=bboxes, box_osd=[], img_yichangs=[])
    state_end, _, bboxes_tag_end = disconnector_state(img_tag_end, img_opens, img_closes, box_state=bboxes, box_osd=[], img_yichangs=[])
    state = state_start + "-" + state_end
    print("final state:", state)

    out_data["data"] = {"result": state_map[state]["name"]}

    ## 保存结果
    for i in range(2):
        b1 = bboxes[0]; b2 = bboxes[1]
        bt1 = bboxes_tag_start[0]; bt2 = bboxes_tag_start[1]
        bt3 = bboxes_tag_end[0]; bt4 = bboxes_tag_end[1]
        cv2.rectangle(img_open, (b1[0], b1[1]), (b1[2], b1[3]), state_map["分-分"]["color"][0], thickness=2)
        cv2.rectangle(img_open, (b2[0], b2[1]), (b2[2], b2[3]), state_map["分-分"]["color"][0], thickness=2)
        cv2.rectangle(img_close, (b1[0], b1[1]), (b1[2], b1[3]), state_map["合-合"]["color"][0], thickness=2)
        cv2.rectangle(img_close, (b2[0], b2[1]), (b2[2], b2[3]), state_map["合-合"]["color"][0], thickness=2)
        cv2.rectangle(img_tag_start, (bt1[0], bt1[1]), (bt1[2], bt1[3]), state_map[state]["color"][0], thickness=2)
        cv2.rectangle(img_tag_start, (bt2[0], bt2[1]), (bt2[2], bt2[3]), state_map[state]["color"][1], thickness=2)
        cv2.rectangle(img_tag_end, (bt3[0], bt3[1]), (bt3[2], bt3[3]), state_map[state]["color"][0], thickness=2)
        cv2.rectangle(img_tag_end, (bt4[0], bt4[1]), (bt4[2], bt4[3]), state_map[state]["color"][1], thickness=2)
    img_tag_start = img_chinese(img_tag_start, state_map[state]["name"], (10, 50), color=state_map[state]["color"][0], size=40)
    img_tag_end = img_chinese(img_tag_end, state_map[state]["name"], (10, 50), color=state_map[state]["color"][0], size=40)

    cv2.imwrite(os.path.join(save_path, TIME_START + "img_open_cfg.jpg"), img_open)
    cv2.imwrite(os.path.join(save_path, TIME_START + "img_close_cfg.jpg"), img_close)
    cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag_start.jpg"), img_tag_start)
    cv2.imwrite(os.path.join(save_path, TIME_START + "img_tag_end.jpg"), img_tag_end)

    f = open(os.path.join(save_path, TIME_START + "output_data.json"),"w",encoding='utf-8')
    json.dump(out_data, f, indent=2)
    f.close()

    out_data["img_result"] = img2base64(img_tag_end)

    return out_data

if __name__ == '__main__':
    input_data = {
        "video_path": "/export/patrolservice/VIDEO/0011_normal.mp4",
        "type": "disconnector_video"
    }
    start = time.time()
    out_data = inspection_disconnector_video(input_data)
    print("spend time:", time.time() - start)
    for s in out_data:
        if s != "img_result":
            print(s,":",out_data[s])

