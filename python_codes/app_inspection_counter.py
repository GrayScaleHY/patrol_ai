import os
import time
import cv2
import json
from lib_image_ops import base642img, img2base64
from lib_inference_yolov5 import load_yolov5_model, inference_yolov5


yolov5_meter = load_yolov5_model("/data/inspection/yolov5/meter.pt") # 加载仪表yolov5模型
yolov5_counter= load_yolov5_model("/data/inspection/yolov5/counter.pt") # 加载记数yolov5模型


def inspection_counter(input_data):
    """
    动作次数数字识别。
    """
    ## 初始化输入输出信息。
    TIME_START = time.strftime("%m-%d-%H-%M-%S") 
    save_path = os.path.join("inspection_result/counter", TIME_START)
    os.makedirs(save_path, exist_ok=True)

    out_data = {"code": 0, "data":{}, "msg": "Success request counter"} # 初始化out_data

    if input_data["type"] != "counter":
        out_data["msg"] = out_data["msg"] + "Type isn't counter; "
        return out_data

    img = base642img(input_data["image"]) # base64格式转numpy格式

    ## 可视化输入信息
    f = open(os.path.join(save_path, "input_data.json"), "w")
    json.dump(input_data, f, ensure_ascii=False)  # 保存输入信息json文件
    f.close()
    cv2.imwrite(os.path.join(save_path, "img_ref.jpg"), img) # 将输入图片可视化

    ## 表盘坐标
    bbox_meters = inference_yolov5(yolov5_meter, img, resize=640)
    if len(bbox_meters) == 0:
        meter_coor = [0.0, 0.0, float(img.shape[1]), float(img.shape[0])]
    ## 只保留应该分值最大的表盘
    scores = [a['score'] for a in bbox_meters]
    rank = [index for index,value in sorted(list(enumerate(scores)),key=lambda x:x[1])]
    meter_coor = bbox_meters[rank[-1]]["coor"]

    m_x = meter_coor[0]; m_y = meter_coor[1]
    img_meter = img[int(meter_coor[1]): int(meter_coor[3]), int(meter_coor[0]): int(meter_coor[2])]

    bbox_counters = inference_yolov5(yolov5_counter, img_meter, resize=640) # 识别数字

    ## 画出表盘
    cv2.rectangle(img, (int(meter_coor[0]), int(meter_coor[1])),
                        (int(meter_coor[2]), int(meter_coor[3])), (0, 0, 255), thickness=2)
    cv2.putText(img, "meter", (int(meter_coor[0])-5, int(meter_coor[1])-5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), thickness=2)

    if len(bbox_counters) == 0:
        out_data["msg"] = out_data["msg"] + "; Not find counter"
        cv2.imwrite(os.path.join(save_path, "img_result.jpg"), img)
        return out_data
    
    ## 根据从左到右的规则对bbox_digitals的存放排序
    l = [a['coor'][0] for a in bbox_counters]
    rank = [index for index,value in sorted(list(enumerate(l)),key=lambda x:x[1])]

    ## 将vals和bboxes添加进out_data
    vals = []; bboxes = []
    for i in rank:
        vals.append(int(bbox_counters[i]['label']))
        coor = bbox_counters[i]['coor']
        coor = [coor[0]+m_x, coor[1]+m_y, coor[2]+m_x, coor[3]+m_y]
        bboxes.append(coor)

    out_data['data'] = {"type": "counter", "values": vals, "bboxes": bboxes}

    ## 可视化计算结果
    f = open(os.path.join(save_path, "out_data.json"), "w")
    json.dump(out_data, f, ensure_ascii=False)  # 保存输入信息json文件
    f.close()
    for i in range(len(out_data['data']["values"])):
        coor = out_data['data']["bboxes"][i]
        label = str(out_data['data']["values"][i])
        cv2.rectangle(img, (int(coor[0]), int(coor[1])),
                        (int(coor[2]), int(coor[3])), (0, 0, 255), thickness=2)
        cv2.putText(img, label, (int(coor[0])-5, int(coor[1])-5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), thickness=2)
    cv2.imwrite(os.path.join(save_path, "img_result.jpg"), img)

    ## 输出可视化结果的图片。
    out_data["img_result"] = img2base64(img)

    return out_data

if __name__ == '__main__':
    from lib_image_ops import img2base64
    img_file = "images/16316064882021.png"
    img_base64 = img2base64(cv2.imread(img_file))
    input_data = {"image": img_base64, "config":[], "type": "counter"}
    out_data = inspection_counter(input_data)
    print(out_data)