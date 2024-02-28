import cv2
import numpy as np
import math
from lib_help_base import GetInputData, creat_img_result
from lib_image_ops import img2base64, img_chinese
from lib_inference_bit import load_bit_model, inference_bit
from lib_img_registration import registration, correct_offset

bit_model = load_bit_model("/data/PatrolAi/bit_cd/bit_cd.pt")

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def process_binary_img(img_diff):
    """
    对二值图做后处理，例如，去除零星的点
    args:
        img_diff: 二值图
    return:
        img_diff: 处理后的二值图
    """
    H, W = img_diff.shape

    # 去除轮廓面积小的点
    contours, hierarchy = cv2.findContours(
        img_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    areas = [cv2.contourArea(c) for c in contours]
    if len(areas) < 2:
        return img_diff
    pts = [cv2.boundingRect(c)[:2] for c in contours]
    ind_max_area = areas.index(max(areas))
    max_area = areas[ind_max_area]
    if max_area < 2:
        return img_diff
    max_pt = pts[ind_max_area]

    d_max = math.sqrt(H ** 2 + W ** 2)
    at = [0, 0.4]  # 面积范围
    contours_f = []
    for i in range(len(areas)):
        d = math.sqrt((max_pt[0]-pts[i][0]) ** 2 + (max_pt[1]-pts[i][1]) ** 2)
        if areas[i] / max_area > at[0] + (at[1] - at[0]) * ((d / d_max) ** 2):
            contours_f.append(contours[i])

    img_diff = cv2.drawContours(np.zeros_like(
        img_diff), contours_f, -1, 255, -1)

    # 对差异性图片进行腐蚀操作，去除零星的点。
    max_rect = cv2.boundingRect(contours[ind_max_area])
    rate = min(max_rect[2:]) / min(H, W)
    erode_it = 1 + int(4 * rate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,5))
    kernel = np.asarray(((0, 0, 1, 0, 0), (0, 1, 1, 1, 0), (1, 1,
                        1, 1, 1), (0, 1, 1, 1, 0), (0, 0, 1, 0, 0)), dtype=np.uint8)
    img_diff = cv2.erode(img_diff, kernel, iterations=erode_it)
    # cv2.imwrite("test1/tag_diff_erode.jpg",img_diff)

    return img_diff

def diff_bit(img_ref, img_tag, resize=512):
    """
    使用bit孪生网络做判别
    """
    H, W = img_tag.shape[:2]
    
    # bit模型推理
    img_diff_raw = inference_bit(bit_model, img_tag, img_ref, resize=resize)

    # 对推理结果mask做腐蚀等后处理
    img_diff = process_binary_img(img_diff_raw)

    # 用轮廓外接矩形作为差异区域
    index_255 = np.where(img_diff == 255)
    index_255 = [a for a in index_255 if len(a) > 1]
    if len(index_255) > 1:
        ymin = max(0, min(index_255[0])-3)
        xmin = max(0, min(index_255[1])-3)
        ymax = min(img_diff.shape[0], max(index_255[0])+3)
        xmax = min(img_diff.shape[1], max(index_255[1])+3)
        tag_diff = [int(xmin), int(ymin), int(xmax), int(ymax)]

        # 最大框不能超过总图片面积的0.6
        h, w = img_diff.shape[:2]
        dif_area = (tag_diff[2] - tag_diff[0]) * (tag_diff[3] - tag_diff[1])
        if dif_area / (h * w) > 0.6:
            tag_diff = []

    else:
        tag_diff = []

    # # 将矩形框内缩5%
    # if len(tag_diff) > 1:
    #     td = tag_diff
    #     in_ = int(min(td[2]-td[0], td[3]-td[1]) * 0.05)
    #     xmin = td[0] if td[0] <= int(resize / 50) else td[0] + in_
    #     xmax = td[2] if td[2] >= int(resize - resize / 50) else td[2] - in_
    #     ymin = td[1] if td[1] <= int(resize / 50) else td[1] + in_
    #     ymax = td[3] if td[3] >= int(resize - resize / 50) else td[3] - in_
    #     tag_diff = [xmin, ymin, xmax, ymax]

    if len(tag_diff) < 1:
        return tag_diff

    rw = W / resize; rh = H / resize
    tag_diff = [int(tag_diff[0]*rw), int(tag_diff[1]*rh),
                int(tag_diff[2]*rw), int(tag_diff[3]*rh)]

    return tag_diff

def inspection_identify_defect(input_data):
    """
    yolov5的目标检测推理。
    """

    # 提取输入请求信息
    DATA = GetInputData(input_data)
    checkpoint = DATA.checkpoint
    an_type = DATA.type
    img_tag = DATA.img_tag
    img_ref = DATA.img_ref

    ## 初始化输入输出信息。
    out_data = {"code": 1, "data":{}, "img_result": input_data["image"], "msg": "Success request object detect; "} # 初始化out_data
    img_tag_ = img_tag.copy()

    # 画上点位名称
    img_tag_ = img_tag.copy()
    img_tag_ = img_chinese(img_tag_, an_type + "_" + checkpoint , (10, 100), color=(255, 0, 0), size=30)

    if img_ref is None:
        out_data["msg"] = out_data["msg"] + "; img_ref not exist;"
        out_data["code"] = 1
        img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=60)
        out_data["img_result"] = creat_img_result(input_data, img_tag_) # 返回结果图
        return out_data

    ## 将两张图片对齐
    M = registration(img_ref, img_tag)
    img_ref, cut = correct_offset(img_ref, M, b=True)
    img_tag = img_tag[cut[1]:cut[3], cut[0]:cut[2], :]
    img_ref = img_ref[cut[1]:cut[3], cut[0]:cut[2], :]

    ## 检测差异区域
    cut_diff = diff_bit(img_ref, img_tag)  

    ## 还原回原图对于的框
    if len(cut_diff) > 0:
        tag_diff = [cut_diff[0] + cut[0], cut_diff[1] +
                    cut[1], cut_diff[2] + cut[0], cut_diff[3] + cut[1]]
    else:
        tag_diff = cut_diff

    if len(tag_diff) == 0:
        img_tag_ = img_chinese(img_tag_, "正常", (20,10), (0, 255, 0), size=40)
        out_data["data"] = {"no_roi": [{"label": "0", "bbox":[]}]}
        out_data["code"] = 0
    else:
        rec = [int(i) for i in tag_diff]
        cv2.rectangle(img_tag_, (int(rec[0]), int(rec[1])),(int(rec[2]), int(rec[3])), (0,0,255), thickness=2)
        img_tag_ = img_chinese(img_tag_, "异常", (int(rec[0])+10, int(rec[1])+20), (0,0,255), size=40)
        out_data["data"] = {"no_roi": [{"label": "1", "bbox":rec}]}
        out_data["code"] = 1

    ## 输出可视化结果的图片。
    img_tag_ = img_chinese(img_tag_, out_data["msg"], (10, 70), color=(255, 0, 0), size=60)

    out_data["img_result"] = creat_img_result(input_data, img_tag_) # 返回结果图

    return out_data

if __name__ == '__main__':
    import time
    from lib_help_base import get_save_head, save_input_data, save_output_data
    ref_file = "/data/PatrolAi/result_patrol/0002_normal.jpg"
    tag_file = "/data/PatrolAi/result_patrol/0002_1.jpg"

    img_tag = img2base64(cv2.imread(tag_file))
    img_ref = img2base64(cv2.imread(ref_file))

    input_data = {"image": img_tag, "config":{"img_ref": img_ref}, "type": "identify_defect"}

    start = time.time()
    out_data = inspection_identify_defect(input_data)
    print(time.time() - start)

    save_dir, name_head = get_save_head(input_data)
    save_input_data(input_data, save_dir, name_head, draw_img=True)
    save_output_data(out_data, save_dir, name_head)
    print("inspection_qrcode result:")
    print("-----------------------------------------------")
    for s in out_data:
        if s != "img_result":
            print(s,":",out_data[s])
    print("----------------------------------------------")


