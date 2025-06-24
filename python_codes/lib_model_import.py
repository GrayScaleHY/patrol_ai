import os.path

from lib_inference_yolov8 import load_yolov8_model
from config_model_list import function_name_list, model_dict,osd_choice
from lib_help_base import GetInputData

model_path = "/data/PatrolAi/yolov8/"

if not {"digital", "counter", "level_gauge", "shuzi_video"}.isdisjoint(set(function_name_list)):
    yolov8_crop = load_yolov8_model(os.path.join(model_path, "jishu_crop.pt"))  # 计数表寻框模型
    yolov8_rec = load_yolov8_model(os.path.join(model_path, "jishu_rec.pt"))  # 计数表数字识别模型
if "level_gauge" in function_name_list:
    yolov8_yeweiji = load_yolov8_model(os.path.join(model_path, "yeweiji.pt"))  # 液位计
if "pointer" in function_name_list:
    yolov8_meter = load_yolov8_model(os.path.join(model_path, "meter.pt"))  # 表记检测
    yolov8_pointer = load_yolov8_model(os.path.join(model_path, "pointer.pt"))  # 指针分割
if "led_color" in function_name_list:
    yolov8_led_color = load_yolov8_model(os.path.join(model_path, "led.pt"))  # led灯颜色状态模型
if not {"led", "led_video", "pressplate", "air_switch", "fanpaiqi", "rotary_switch", "door"}.isdisjoint(
        set(function_name_list)):
    yolov8_ErCiSheBei = load_yolov8_model(os.path.join(model_path, "ErCiSheBei.pt"))  ## 二次设备状态
if "disconnector_notemp" in function_name_list:
    yolov8_daozha = load_yolov8_model(os.path.join(model_path, "daozha_v5detect.pt"))  # 加载刀闸模型
if "disconnector_texie" in function_name_list:
    yolov8_dztx = load_yolov8_model(os.path.join(model_path, "daozha_texie.pt"))  # 刀闸特写
if "rec_defect" in function_name_list:
    yolov8_rec_defect = load_yolov8_model(os.path.join(model_path, "rec_defect.pt"))  # 送检18类缺陷,x6模型
if "sticker" in function_name_list:
    yolov8_sticker = load_yolov8_model(os.path.join(model_path, "sticker.pt"))  # 检修状态标签
if "key" in function_name_list:
    yolov8_count = load_yolov8_model(os.path.join(model_path, "count.pt"))  # 大电流端子借用钥匙计数功能
if "fire_smoke" in function_name_list:
    yolov8_fire_smoke = load_yolov8_model(os.path.join(model_path, "fire_smoke.pt"))  # 烟火
if "helmet" in function_name_list:
    yolov8_helmet = load_yolov8_model(os.path.join(model_path, "helmet.pt"))  # 安全帽
if "person" in function_name_list:
    yolov8_coco = load_yolov8_model(os.path.join(model_path, "coco.pt"))  # coco人员模型
if "biaoshipai" in function_name_list:
    yolov8_biaoshipai = load_yolov8_model(os.path.join(model_path, "biaoshipai.pt"))
if osd_choice:
    osd_model=load_yolov8_model(os.path.join(model_path,"osd.pt")) #osd检测模型

# yolov8_action = load_yolov8_model("/data/PatrolAi/yolov8/action.pt") # 动作识别，倒地



#模型选用
def model_load(an_type):
    yolov8_model = model_dict[an_type]
    if type(yolov8_model) == list:
        yolov8_model = [eval(item) for item in yolov8_model]
    else:
        yolov8_model = eval(yolov8_model)
    return yolov8_model