# 功能选择清单
function_name_list = [
    "digital",  # 液晶屏数字识别（带小数位）
    "counter",  # 计数数字识别（不带小数位）
    "level_gauge",  # 液位计
    "pointer",  # 表计识别
    "led_color",  # 指示灯颜色
    "led",  # 指示灯状态
    "led_video",  # 视频识别指示灯
    "pressplate",  # 压板状态
    "air_switch",  # 空开状态
    "fanpaiqi",  # 翻牌器状态
    "rotary_switch",  # 旋钮开关状态
    "disconnector_notemp",  # 刀闸状态
    "rec_defect",  # 缺陷识别
    #    "fire_smoke",  # 烟雾识别
    #    "disconnector_texie",     # 刀闸特写
    #    "door",                   # 箱门闭合状态
    #    "key",                    # 钥匙计数
    #    "shuzi_video",            # 视频数字识别
    #    "sticker",                # 维修状态标签识别
    #    "helmet",                # 安全帽识别
    #    "biaoshipai",             # ？
    #    "person",                 # 人员
]

# 模型阈值
model_threshold_dict = {
    "rec_defect": 0.8,
    "digital": 0.25,
    "counter": 0.25,
    "shuzi_video": 0.25,
    "level_gauge": 0.25,
    "pointer": 0.25,
    "led_color": 0.3,
    "led": 0.3,
    "led_video": 0.3,
    "pressplate": 0.3,
    "air_switch": 0.3,
    "fanpaiqi": 0.3,
    "rotary_switch": 0.3,
    "door": 0.3,
    "disconnector_notemp": 0.3,
    "disconnector_texie": 0.3,
    "sticker": 0.3,
    "key": 0.3,
    "fire_smoke": 0.3,
    "helmet": 0.3,
    "person": 0.3,
    "biaoshipai": 0.3,
}

# 模型标签筛选
model_label_dict = {
    "disconnector_notemp": ["he", "fen", "budaowei"],
    "disconnector_texie": ["he", "fen", "budaowei"],
    "person": ["person"],
    "pressplate": ["kgg_ybh", "kgg_ybf", "byyb"],
    "air_switch": ["kqkg_hz", "kqkg_fz"],
    "led": ["zsd_l", "zsd_m"],
    "fanpaiqi": ["fpq_h", "fpq_f", "fpq_jd"],
    "rotary_switch": ["xnkg_s", "xnkg_zs", "xnkg_ys", "xnkg_z", "xnkg_y", "xnkg_yx", "xnkg_x", "xnkg_zx"],
    "door": ["xmbhyc", "xmbhzc"],
    "key": ["ys"],
}

# 功能模型导入列表
model_dict = {
    "digital": ["yolov8_crop", "yolov8_rec"],
    "counter": ["yolov8_crop", "yolov8_rec"],
    "shuzi_video": ["yolov8_crop", "yolov8_rec"],
    "level_gauge": ["yolov8_crop", "yolov8_rec", "yolov8_yeweiji"],
    "pointer": ["yolov8_meter", "yolov8_pointer"],
    "led_color": "yolov8_led_color",
    "led": "yolov8_ErCiSheBei",
    "led_video": "yolov8_ErCiSheBei",
    "pressplate": "yolov8_ErCiSheBei",
    "air_switch": "yolov8_ErCiSheBei",
    "fanpaiqi": "yolov8_ErCiSheBei",
    "rotary_switch": "yolov8_ErCiSheBei",
    "door": "yolov8_ErCiSheBei",
    "disconnector_notemp": "yolov8_daozha",
    "disconnector_texie": "yolov8_dztx",
    "rec_defect": "yolov8_rec_defect",
    "sticker": "yolov8_sticker",
    "key": "yolov8_count",
    "fire_smoke": "yolov8_fire_smoke",
    "helmet": "yolov8_helmet",
    "person": "yolov8_coco",
    "biaoshipai": "yolov8_biaoshipai",
}

model_type_dict = {
    "led_color": "led",
    "led": "ErCiSheBei",
    "pressplate": "ErCiSheBei",
    "air_switch": "ErCiSheBei",
    "fanpaiqi": "ErCiSheBei",
    "rotary_switch": "ErCiSheBei",
    "door": "ErCiSheBei",
    "disconnector_notemp": "disconnector_texie",
    "disconnector_texie": "disconnector_texie",
    "rec_defect": "rec_defect",
    "key": "ErCiSheBei",
    "fire_smoke": "fire_smoke",
    "helmet": "helmet",
    "person": "coco",
    "biaoshipai": "biaoshipai",
}

# 纠偏算法优先级排序
registration_model_list = {
    1: "eflotr",
    2: "lightglue",
    3: "loftr",
}
