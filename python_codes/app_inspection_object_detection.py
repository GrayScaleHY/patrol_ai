import os
import time
import cv2
import json
from lib_image_ops import base642img, img2base64
from lib_inference_yolov5 import load_yolov5_model, inference_yolov5
from lib_help_base import color_list

# yolov5_meter = load_yolov5_model("/data/inspection/yolov5/meter.pt") # 表盘
yolov5_air_switch = load_yolov5_model("/data/inspection/yolov5/air_switch.pt") # 空气开关
yolov5_fire_smoke = load_yolov5_model("/data/inspection/yolov5/fire_smoke.pt") # 烟火
yolov5_led = load_yolov5_model("/data/inspection/yolov5/led.pt") # led灯
yolov5_pressplate = load_yolov5_model("/data/inspection/yolov5/pressplate.pt") # 压板
# yolov5_helmet = load_yolov5_model("/data/inspection/yolov5/helmet.pt") # 安全帽
yolov5_fanpaiqi = load_yolov5_model("/data/inspection/yolov5/fanpaiqi.pt") # 翻拍器

def inspection_object_detection(input_data):
    """
    yolov5的目标检测推理。
    """
    ## 初始化输入输出信息。
    out_data = {"code": 0, "data":[], "img_result": "image", "msg": "Success request object detect; "} # 初始化out_data

    ## 可视化输入信息
    TIME_START = time.strftime("%m-%d-%H-%M-%S") 
    save_path = os.path.join("inspection_result", input_data["type"], TIME_START)
    os.makedirs(save_path, exist_ok=True)
    f = open(os.path.join(save_path, "input_data.json"), "w")
    json.dump(input_data, f, ensure_ascii=False)  # 保存输入信息json文件
    f.close()
    img = base642img(input_data["image"])
    cv2.imwrite(os.path.join(save_path, "img_ref.jpg"), img) # 将输入图片可视化

    ## 选择模型
    if input_data["type"] == "pressplate": # ["air_switch", "fire_smoke", "led", "pressplate"]:
        yolov5_model = yolov5_pressplate
    # elif input_data["type"] == "meter":
    #     yolov5_model = yolov5_meter
    elif input_data["type"] == "air_switch":
        yolov5_model = yolov5_air_switch
    elif input_data["type"] == "fire_smoke":
        yolov5_model = yolov5_fire_smoke
    elif input_data["type"] == "led":
        yolov5_model = yolov5_led
    # elif input_data["type"] == "helmet":
    #     yolov5_model = yolov5_helmet
    elif input_data["type"] == "yolov5_fanpaiqi":
        yolov5_model = yolov5_fanpaiqi
    else:
        out_data["msg"] = out_data["msg"] + "Type isn't object; "
        return out_data

    ## 生成目标检测信息
    bboxes = inference_yolov5(yolov5_model, img, resize=640) # inference
    if len(bboxes) == 0: #没有检测到目标
        out_data["msg"] = out_data["msg"] + "; Not find object"
        return out_data
    for bbox in bboxes:
        cfg = {"type": bbox["label"], "bbox": bbox["coor"]}
        out_data["data"].append(cfg)
    
    ## labels 和 color的对应关系
    labels = yolov5_model.module.names if hasattr(yolov5_model, 'module') else yolov5_model.names
    colors = color_list(len(labels))
    color_dict = {}
    for i, label in enumerate(labels):
        color_dict[label] = colors[i]

    ## 可视化计算结果
    f = open(os.path.join(save_path, "out_data.json"), "w")
    json.dump(out_data, f, ensure_ascii=False, indent=2)  # 保存输入信息json文件
    f.close()
    for bbox in bboxes:
        coor = bbox["coor"]; label = bbox["label"]
        s = (coor[2] - coor[0]) / 200 # 根据框子大小决定字号和线条粗细。
        cv2.rectangle(img, (int(coor[0]), int(coor[1])),
                        (int(coor[2]), int(coor[3])), color_dict[label], thickness=round(s*2))
        cv2.putText(img, label, (int(coor[0])-5, int(coor[1])-5),
                    cv2.FONT_HERSHEY_COMPLEX, s, color_dict[label], thickness=round(s*2))
    cv2.imwrite(os.path.join(save_path, "img_result.jpg"), img)

    ## 输出可视化结果的图片。
    out_data["img_result"] = img2base64(img)

    return out_data


if __name__ == '__main__':
    from lib_image_ops import img2base64
    img_file = "/data/yolov5/pressplate/images/train/IMG_4197_360.jpg"
    img = cv2.imread(img_file)
    img_base64 = img2base64(img)
    input_data = {"image": img_base64, "config":[], "type": "pressplate"}
    out_data = inspection_object_detection(input_data)
    print("inspection_object_detection result:")
    print("-----------------------------------------------")
    for s in out_data:
        if s != "img_result":
            print(s,":",out_data[s])
    print("----------------------------------------------")
    



