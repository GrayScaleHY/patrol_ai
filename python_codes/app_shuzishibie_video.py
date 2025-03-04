import os
import cv2
import time
import json
from lib_image_ops import  img_chinese
from lib_inference_yolov8 import  inference_yolov8
from lib_rcnn_ops import check_iou
from lib_img_registration import  roi_registration
from lib_help_base import GetInputData, creat_img_result,  reg_crop
from app_yejingpingshuzishibie import get_dp, img_fill, dp_append, yolov8_jishukuang, yolov8_jishushibie


def inspection_pic_fromvideo(frame, img_ref, roi,dp):
    '''
        frame_dict为两层字典结构：
        {   "code":0  #判断非空
            “name1”：{"phase":[phase,number],
                     "coor":[1,2,3,4],
                    }，
            “name2”:{..}
        }
    '''
    frame_dict={"code":1}
    phase_label=["Ia","Ib","Ic","Ua","Ub","Uc"]
    yolo_crop, yolo_rec = yolov8_jishukuang, yolov8_jishushibie

    #目标框映射获取
    roi_tag_list, _ = roi_registration(img_ref, frame, roi)

    #一阶段识别，判断是否在roi_tag中
    bbox_cfg = inference_yolov8(yolo_crop, frame)
    # 未检测到目标,按目标框检测
    if len(bbox_cfg) < 1:
        coor_list = []
        for roi_tag_item in roi_tag_list:
            coor_list.append([int(item) for item in roi_tag_list[roi_tag_item]])
    else:
        coor_list = [item['coor'] for item in bbox_cfg]
    #bboxes_list_sort = sorted(coor_list, key=lambda x: x[-1], reverse=False)
    roi_name = "no_roi"
    roi_name_dict = {}
    for coor in coor_list:
        roi_dict = {}
        if len(roi_tag_list) == 0:
            roi_dict["bbox"] = coor
            mark = True
        else:
            # 去除roi框外，不做识别
            x_middle = (coor[0] + coor[2]) / 2
            y_middle = (coor[1] + coor[3]) / 2
            mark = False
            for roi_tag in roi_tag_list:
                if (x_middle < roi_tag_list[roi_tag][2] and x_middle > roi_tag_list[roi_tag][0]) and (
                        y_middle < roi_tag_list[roi_tag][3] and y_middle > roi_tag_list[roi_tag][1]):
                    mark = True
                    roi_dict["bbox"] = coor
                    roi_name = roi_tag
                    break
        if not mark:
            continue
        # 640*640填充
        img_empty = img_fill(frame, coor[0], coor[1], coor[2], coor[3], 640)
        # 二次识别
        bbox_cfg_result = inference_yolov8(yolo_rec, img_empty)
        bbox_cfg_result = check_iou(bbox_cfg_result, 0.2)
        if len(bbox_cfg_result)==[]:
            continue
        roi_dict["value"] = [[item['label'], item['coor']] for item in bbox_cfg_result]
        try:
            roi_name_dict[roi_name].append(roi_dict) #{"name":[{"value":[item['label'], item['coor']] ,"bbox":coor},....]}
        except:
            roi_name_dict[roi_name]=[]
            roi_name_dict[roi_name].append(roi_dict)

    #roi_name_dict转化为frame_dict

    for roi_name in roi_tag_list:
        frame_dict[roi_name] = {}
        # 无框，下一个roi_name
        try:
            label_list=roi_name_dict[roi_name]
        except:
            continue

        #单框，区分phase、数字；双框以上，左phase，右数字，其余框忽略
        if len(label_list)==1:  #[{"value":[[item['label'], item['coor'],...]] ,"bbox":coor},....]
            value_list = sorted(label_list[0]["value"], key=lambda x: x[1][0], reverse=False)
            for item in value_list:  #[[item['label'], item['coor']],...]
                if item[0] in phase_label:
                    phase=item[0]
                    value_list=[k[0] for k in value_list if k[0].isdigit()]
                    if len(value_list) == 0:
                        continue
                    if dp != 0:
                        value_list = dp_append(value_list, dp)
                    value="".join(value_list)
                    frame_dict[roi_name]["phase"]=[phase,value]
                    frame_dict[roi_name]["coor"]=label_list[0]["bbox"]
                    frame_dict["code"]=0
                    break
        else:
            # 按横坐标排序组合结果
            label_list = sorted(label_list, key=lambda x: x['bbox'][0], reverse=False)  #[{"value":[item['label'], item['coor']] ,"bbox":coor},....]
            phase_list=[k[0] for k in label_list[0]["value"] if k[0] in phase_label]
            if phase_list == []:
                continue
            phase="".join(phase_list)
            value_list = label_list[1]['value']    #[[item['label'], item['coor']],...]
            value_list = sorted(value_list, key=lambda x: x[1][0], reverse=False)
            value_list = [k[0] for k in value_list if k[0].isdigit()]
            if len(value_list)==0:
                continue
            if dp != 0:
                value_list = dp_append(value_list, dp)
            value = "".join(value_list)
            frame_dict[roi_name]["phase"] =[phase,value]
            frame_dict[roi_name]["coor"] = label_list[1]["bbox"]
            frame_dict["code"] = 0
    return frame_dict


def inspection_digital_rec_video(input_data):
    out_data = {"code": 0, "data": {}, "img_result": "", "msg": "Request;"}  # 初始化输出信息

    ## 提取输入请求信息
    DATA = GetInputData(input_data)
    checkpoint = DATA.checkpoint
    an_type = DATA.type
    roi = DATA.roi
    reg_box = DATA.regbox
    img_ref = DATA.img_ref
    video_path = DATA.video_path
    dp = get_dp(DATA.config)

    if not os.path.exists(video_path):
        out_data["msg"] = out_data["msg"] + video_path + " not exists !"
        out_data["code"] = 1
        return out_data

    if an_type != "shuzi_video":
        img_tag_ = img_ref.copy()
        img_tag_ = img_chinese(img_tag_, checkpoint + input_data["type"], (10, 10), color=(255, 0, 0), size=60)
        out_data["msg"] = out_data["msg"] + " wrong type!"
        out_data["img_result"] = img2base64(img_tag_)
        out_data["code"] = 1
        return out_data

    # img_ref截取regbox区域用于特征匹配
    if reg_box and len(reg_box) != 0:
        img_ref = reg_crop(img_ref, *reg_box)

    # 读取视频，获取每帧结果
    step = 1
    count = 0
    if_draw=True
    cap = cv2.VideoCapture(video_path)  ## 建立视频对象
    video_resultdict = {}
    while (cap.isOpened()):
        ret, frame = cap.read()  # 逐帧读取
        if frame is not None and count % step == 0:
            # 识别结果，返回结果列表
            frame_dict = inspection_pic_fromvideo(frame,img_ref,roi,dp)
            if frame_dict["code"]==1:
                count += 1
                frame_last = frame
                continue

            #展示图
            if if_draw:
                frame_ = frame.copy()
                for item in frame_dict:
                    try:
                        coor=frame_dict[item]["coor"]
                    except:
                        continue
                    s = (coor[2] - coor[0]) / 50  # 根据框子大小决定字号和线条粗细。
                    cv2.putText(frame_,":".join(frame_dict[item]["phase"]), (coor[2], coor[3]), cv2.FONT_HERSHEY_SIMPLEX, round(s), (0, 255, 0),
                                thickness=round(s * 2))
                    cv2.rectangle(frame_, (int(coor[0]), int(coor[1])), (int(coor[2]), int(coor[3])), (255, 0, 255),
                                  thickness=1)
                if_draw=False
            # 总数累积
            for roi_name in frame_dict:
                if roi_name == "code":
                    continue
                if frame_dict[roi_name]=={}:
                    video_resultdict[roi_name]={}
                    continue
                try:
                    if frame_dict[roi_name]["phase"][0] in video_resultdict[roi_name]:
                        video_resultdict[roi_name][frame_dict[roi_name]["phase"][0]][0]+=float(frame_dict[roi_name]["phase"][1])
                        video_resultdict[roi_name][frame_dict[roi_name]["phase"][0]][1]+=1
                    else:
                        video_resultdict[roi_name][frame_dict[roi_name]["phase"][0]]=[float(frame_dict[roi_name]["phase"][1]),1]
                except:
                    video_resultdict[roi_name]= {frame_dict[roi_name]["phase"][0]:[float(frame_dict[roi_name]["phase"][1]),1]}

        '''
        frame_dict为两层字典结构：
        {"code":0
        “name1”：{"phase": [phase1,number1],
                 "coor": [1,2,3,4]],
                 }，
        “name2”:{..}
        }
        video_resultdict:
        {
        "name":{"phase1":[sum,count],
                "phase2":[sum,count],
                }
        }
        '''
        count += 1
        if not ret:
            break
        frame_last=frame
    cap.release()

    if video_resultdict=={}:
        frame = img_chinese(frame_last, checkpoint + input_data["type"], (10, 10), color=(255, 0, 0), size=60)
        out_data["msg"] = out_data["msg"] + " Cant found!"
        out_data["img_result"] = creat_img_result(input_data, frame)
        out_data["code"] = 1
        return out_data

    out_data_dict={}
    for roi_name in video_resultdict:
        out_data_dict[roi_name]=[]
        for phase in video_resultdict[roi_name]:
            out_data_dict[roi_name].append(
                {"phase":phase,
                "label":round(video_resultdict[roi_name][phase][0]/video_resultdict[roi_name][phase][1],3),
                "score":0.9}
            )
    out_data['code']=0
    out_data["data"]=out_data_dict
    out_data["img_result"] = creat_img_result(input_data, frame_)  # 返回结果图
    return out_data


if __name__ == '__main__':
    from lib_image_ops import img2base64
    from lib_help_base import get_save_head, save_output_data

    with open("test/dankuang.json", "r", encoding="utf8") as f:
        input_data = json.load(f)
    start = time.time()
    out_data = inspection_digital_rec_video(input_data)
    print("spend time:", time.time() - start)

    print("-----------------------------------------------")
    for s in out_data:
        if s != "img_result":
            print(s, ":", out_data[s])
    print("----------------------------------------------")

    save_dir, name_head = get_save_head(input_data)
    #save_input_data(input_data, save_dir, name_head, draw_img=True)
    save_output_data(out_data, save_dir, name_head)

