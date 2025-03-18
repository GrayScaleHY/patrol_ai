import os
import time
import cv2
from lib_image_ops import base642img, img2base64, img_chinese
from lib_inference_yolov8 import load_yolov8_model, inference_yolov8
from lib_img_registration import roi_registration
import numpy as np
from lib_help_base import GetInputData, is_include

ErCiSheBei_model = load_yolov8_model("/data/PatrolAi/yolov8/ErCiSheBei.pt") ## 二次设备状态

def patrolai_led_video(input_data):
    """
    led灯视频分析，主要用于解决led灯闪烁识别的问题。
    接口形式：
    https://git.utapp.cn/xunshi-ai/json-http-interface/-/wikis/智能巡检_led灯亮灭状态识别_视频分析
    """
    ## 提取输入请求信息
    DATA = GetInputData(input_data)
    checkpoint = DATA.checkpoint; an_type = DATA.type
    video_path = DATA.video_path; img_ref = DATA.img_ref
    roi = DATA.roi
    
    ## 初始化out_data
    out_data = {"code": 0, "data":{}, "img_result": "", "msg": "Request; "}

    if not os.path.exists(video_path):
        out_data["msg"] = out_data["msg"] + video_path + " not exists !"
        out_data["code"] = 1
        return out_data

    ## 模型类型
    labels = ["zsd_l", "zsd_m"]
    model_type = "ErCiSheBei"

    ## 逐帧截取视频
    step = 1
    cap = cv2.VideoCapture(video_path) ## 建立视频对象
    count = 0
    cfgs_v = []
    l_max = -1
    while(cap.isOpened()):
        ret, frame = cap.read() # 逐帧读取
        if frame is not None and count % step == 0:
            img_tag = frame
            cfgs = inference_yolov8(ErCiSheBei_model, img_tag, resize=640, focus_labels=labels, conf_thres=0.3) # inference
            cfgs_v.append(cfgs)

            ## zsd_l目标最多的图片作为展示图
            l_ = len([cfg["label"] for cfg in cfgs if cfg["label"] == "zsd_l"])
            if l_ > l_max:
                img_tag_ = img_tag
                l_max = l_

        count += 1
        if not ret:
            break
    cap.release()

    ## 求出目标图像的感兴趣区域
    roi_tag, _ = roi_registration(img_ref, img_tag, roi)
    for name, c in roi_tag.items():
        cv2.rectangle(img_tag_, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), (255,0,255), thickness=1)
        cv2.putText(img_tag_, name, (int(c[0]), int(c[1])+10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=1)
    
    ## 画上点位名称
    img_tag_ = img_chinese(img_tag_, checkpoint + an_type , (10, 10), color=(255, 0, 0), size=60)

    ## 判断cfgs_v中的bbox是否在roi_tag中
    for name, roi_ in roi_tag.items():
        status = []
        for cfgs in cfgs_v:
            for cfg in cfgs:
                if is_include(cfg["coor"], roi_, srate=0.5):
                    if cfg["label"] == "zsd_l":
                        status.append(0)
                    else:
                        status.append(1)
        
        rate = (sum(status) + 0.1) / (len(status) + 0.1)
        if rate < 0.1:
            final_label = "指示灯亮"
        elif rate > 0.99:
            final_label = "指示灯灭"
        else:
            final_label = "指示灯闪烁"
        
        out_data["data"][name] = [{"label": final_label, "bbox": roi_, "score": 1.0}]
    
    c = roi_
    s = int((c[2] - c[0]) / 6) # 根据框子大小决定字号和线条粗细。
    img_tag_ = img_chinese(img_tag_, final_label, (c[0], c[1]), (0,0,255), size=s)

    if os.path.exists(input_data["video_path"]): 
        out_file = input_data["video_path"][:-4] + "_result.jpg"
        cv2.imwrite(out_file, img_tag_)
        img_result = out_file
    else:
        img_result = img2base64(img_tag_)
    out_data["img_result"] = img_result

    return out_data

if __name__ == '__main__':
    from lib_image_ops import img2base64
    from lib_help_base import get_save_head, save_input_data, save_output_data
    video_path = "/data/PatrolAi/patrol_ai/python_codes/test/led_test.mp4"
    ref_file = "/data/PatrolAi/patrol_ai/python_codes/test/img_ref.jpg"
    # roi = [539, 1020, 763, 1234] # 闪烁
    # roi = [521, 338, 745, 542] # 指示灯灭
    roi = [523, 679, 749, 880] # 指示灯亮
    img_ref = cv2.imread(ref_file)
    H, W = img_ref.shape[:2]
    roi = [roi[0]/W, roi[1]/H, roi[2]/W, roi[3]/H]
    ref_base64 = img2base64(img_ref)
    config = {"img_ref": ref_base64, "bboxes": {"roi": roi}}
    input_data = {"video_path": video_path, "config": config, "type": "led_video"}

    start = time.time()
    out_data = patrolai_led_video(input_data)
    print("spend time:", time.time() - start)

    print("-----------------------------------------------")
    for s in out_data:
        if s != "img_result":
            print(s,":",out_data[s])
    print("----------------------------------------------")

    save_dir, name_head = get_save_head(input_data)
    save_input_data(input_data, save_dir, name_head, draw_img=True)
    save_output_data(out_data, save_dir, name_head)
