import os
import time
import cv2
import json
from lib_image_ops import base642img, img2base64
from lib_inference_yolov5 import load_yolov5_model, inference_yolov5

yolov5_meter = load_yolov5_model("/data/inspection/yolov5/meter.pt") # 加载仪表yolov5模型

def inspection_meter(input_data):
    """
    仪表定位。
    """
    ## 初始化输入输出信息。
    TIME_START = time.strftime("%m-%d-%H-%M-%S") 
    save_path = os.path.join("meter", TIME_START)
    os.makedirs(save_path, exist_ok=True)

    out_data = {"code": 0, "data":[], "msg": "Success request meter"} # 初始化out_data

    if input_data["type"] != "meter":
        out_data["msg"] = out_data["msg"] + "Type isn't meter; "
        return out_data

    img = base642img(input_data["image"]) # base64格式转numpy格式

    ## 可视化输入信息
    f = open(os.path.join(save_path, "input_data.json"), "w")
    json.dump(input_data, f, ensure_ascii=False)  # 保存输入信息json文件
    f.close()
    cv2.imwrite(os.path.join(save_path, "img_ref.jpg"), img) # 将输入图片可视化

    bbox_meters = inference_yolov5(yolov5_meter, img, resize=640) # yolov5模型识别

    if len(bbox_meters) == 0: # 没检测到表盘
        out_data["msg"] = out_data["msg"] + "; Not find meter"
        return out_data

    cfg = {"type": "meter", "bboxes":[]}
    for bbox in bbox_meters:
        cfg["bboxes"].append(bbox["coor"])

    out_data["data"] = cfg

    ## 可视化计算结果
    f = open(os.path.join(save_path, "out_data.json"), "w")
    json.dump(out_data, f, ensure_ascii=False)  # 保存输入信息json文件
    f.close()
    for bbox in bbox_meters:
        coor = bbox["coor"]
        cv2.rectangle(img, (int(coor[0]), int(coor[1])),
                        (int(coor[2]), int(coor[3])), (0, 0, 255), thickness=2)
        cv2.putText(img, "meter", (int(coor[0])-5, int(coor[1])-5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), thickness=2)
    cv2.imwrite(os.path.join(save_path, "img_result.jpg"), img)
    
    return out_data

if __name__ == '__main__':
    from lib_image_ops import img2base64
    img_file = "images/img_ref.jpg"
    img_base64 = img2base64(img_file)
    input_data = {"image": img_base64, "config":[], "type": "meter"}
    out_data = inspection_meter(input_data)
    print(out_data)


