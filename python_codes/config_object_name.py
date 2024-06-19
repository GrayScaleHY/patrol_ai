## 目标检测，若没检测到返回的默认状态
DEFAULT_STATE = {
    "abnormal_code": 1, # 识别异常是的code，若code=0,则验收时不会产生异常
    "led": "指示灯灭", # 指示灯默认状态
    "fanpaiqi": "识别异常", # 翻牌器默认状态
    "disconnector_notemp": "识别异常", # 默认刀闸识别
    "disconnector_texie": "识别异常",  # 刀闸特写识别
    "door": "箱门正常", # 箱门闭合状态
    "air_switch": None, # 空气开关
    "pressplate": None, # 压板
}

jmjs_dict = {
    "sb_bx": {"name": "设备变形", "labels": ["pzqcd", "jyhbx", "drqgd"]},
    "sb_dl": {"name": "设备断裂", "labels": ["jyz_pl", "yxdgsg", "jdyxxsd"]},
    "rq_yw": {"name": "异物入侵", "labels": ["yw_nc", "yw_gkxfw"]},
    "rq_ry": {"name": "越线闯入", "labels": ["person"]},
    "rq_xdw": {"name": "小动物", "labels": ["bird", "cat", "dog", "sheep", "animal"]},
    "js_dm": {"name": "积水", "labels": ["jishui"]},
    # "hzyw": {"name": "设备烟火", "labels": ["hzyw"]},
    # "wcgz": {"name": "未穿长袖工作服", "labels": ["wcgz"]},
    # "wcaqm": {"name": "未正确佩戴安全帽", "labels": ["wcaqm"]},
    # "sly_bjbmyw": {"name": "渗漏油", "labels": ["sly_bjbmyw"]},
    # "sly_dmyw": {"name": "渗漏油", "labels": ["sly_dmyw"]},
}

## 目标检测中各标签的命名
OBJECT_MAP = {
    "ErCiSheBei":{
        'xmbhyc': '箱门异常',
        'xmbhzc': '箱门正常',
        'kgg_ybh': '合',
        'kgg_ybf': '分',
        'kqkg_hz': '合闸',
        'kqkg_fz': '分闸',
        'fpq_h': '翻牌器合',
        'fpq_f': '翻牌器分',
        'fpq_jd': '翻牌器异常',
        'xnkg_s': '上',
        'xnkg_zs': '左上',
        'xnkg_ys': '右上',
        'xnkg_z': '左',
        'ys': '钥匙',
        'zsd_l': "指示灯亮",
        'zsd_m': '指示灯灭'
    },

    "led": {
        "name": "LED灯",
        "led_off": "指示灯_灭",
        "green_dc": "绿灯亮_断开",
        "red_c": "红灯亮_接通",
        "red_off": "红灯亮_接地",
        "red_cn": "红灯亮_储能灯",
        "yellow_cn": "黄灯亮_储能灯",
        "white_cn": "白灯亮_储能灯",
        "green_cn": "绿灯亮_储能灯",
        "yellow_ml": "黄灯亮_米粒灯",
        "white_ml": "白灯亮_米粒灯",
        "green_ml": "绿灯亮_米粒灯",
        "red_ml": "红灯亮_米粒灯",
        "yellow_yx": "黄灯亮_运行灯",
        "white_yx": "白灯亮_运行灯",
        "green_yx": "路灯亮_运行灯",
        "red_yx": "红灯亮_运行灯"
    },
    "rec_defect": {
        'sly_bjbmyw': '部件表面油污',
        'sly_dmyw': '地面油污',
        'pzqcd': '膨胀器冲顶', # '金属膨胀器冲顶破损',
        'jyhbx': '均压环破损变形',
        'drqgd': '电容器鼓肚',
        'jyz_pl': '绝缘子破裂', # '外绝缘裂纹破损'
        'yxdgsg': '引线断股松股', #
        'jdyxxsd': '接地引下线松动',
        'jsxs_ddjt': '金属锈蚀',
        'jsxs_ddyx': '金属锈蚀',
        'jsxs_jdyxx': '金属锈蚀',
        'jsxs_ecjxh': '金属锈蚀',
        'hzyw': '火灾烟雾', # '设备烟火',
        'bmwh': '表面污秽',
        'ws_ywyc': '瓦斯观察窗油位下降',
        'ws_ywzc': '瓦斯观察窗油位正常',
        'yljdq_flow': '流油继电器flow',
        'yljdq_stop': '流油继电器stop',
        'fhz_f': '刀闸指示分',
        'fhz_h': '刀闸指示合',
        'fhz_ztyc': '刀闸指示异常',
        'hxq_gjbs': '呼吸器硅胶变色',
        'hxq_gjzc': '呼吸器硅胶正常',
        'ywzt_yfyc': '油位状态油封异常', # '油封油位异常',
        'ywzt_yfzc': '油封油位正常',
        'hxq_gjtps': '硅胶筒破损',
        'hxq_yfps': '呼吸器油封破损',
        'bj_wkps': '表计外壳破损',
        'bj_bpps': '表计表盘破损',
        'bjzc': '表计正常',
        'bjdsyc_zz': '表计读数异常', # '指针表表计读数异常'
        'bjdsyc_sx': '表计读数异常', # '数显表表计读数异常',
        'bjdsyc_ywj': '表计读数异常', # '油位表计读数异常',
        'bjdsyc_ywc': '表计读数异常', # '油位观察窗异常',
        'bj_bpmh': '表计表盘模糊',
        'gcc_mh': '观察窗模糊',
        'gcc_ps': '观察窗破损',
        'xmbhyc': '箱门闭合异常',
        'xmbhzc': '箱门闭合正常',
        'yw_nc': '异物－鸟巢',
        'yw_gkxfw': '异物－高空悬浮物',
        'kgg_ybf': '压板分',
        'kgg_ybh': '压板合',
        'kk_f': '空开分',
        'kk_h': '空开合',
        'zsd_l': '指示灯亮',
        'zsd_m': '指示灯灭',
        'mbhp': '面板指示异常',
        'wcgz': '未穿工装',
        'gzzc': '工装正常',
        'wcaqm': '未穿安全帽',
        'aqmzc': '安全帽正常',
        'xy': '吸烟'
    },
    "coco": {
        "name": "coco labels",
        "person": "人员", "bicycle": "自行车", "car": "車", "motorcycle": "摩托", "airplane": "飞机", "bus": "公共汽车", "train": "火车", 
        "truck": "卡车", "boat": "小船", "traffic light": "红绿灯", "fire hydrant": "消火栓", "stop sign": "停车标记", 
        "parking meter": "停车记时器", "bench": "法官", "bird": "鸟", "cat": "猫", "dog": "狗", "horse": "馬", "sheep": "綿羊", "cow": "牛", 
        "elephant": "大象", "bear": "熊", "zebra": "斑馬", "giraffe": "长颈鹿", "backpack": "背包", "umbrella": "伞", "handbag": "手提包", 
        "tie": "领带", "suitcase": "手提箱", "frisbee": "飞盘", "skis": "滑雪板", "snowboard": "滑雪板", "sports ball": "体育运动", "kite": "鳶", 
        "baseball bat": "棒球棒", "baseball glove": "棒球手套", "skateboard": "滑板", "surfboard": "冲浪板", "tennis racket": "球拍", "bottle": "瓶子", 
        "wine glass": "酒杯", "cup": "杯子", "fork": "分叉", "knife": "小刀", "spoon": "勺子", "bowl": "碗", "banana": "香蕉", "apple": "苹果", 
        "sandwich": "三明治", "orange": "橙色", "broccoli": "西兰花", "carrot": "胡萝卜", "hot dog": "热狗", "pizza": "比萨", "donut": "甜甜圈", 
        "cake": "蛋糕", "chair": "椅子", "couch": "沙发", "potted plant": "车间", "bed": "床上", "dining table": "餐桌", "toilet": "厕所", "tv": "电视", 
        "laptop": "笔记本", "mouse": "鼠标", "remote": "间接", "keyboard": "键盘", "cell phone": "手机", "microwave": "微波", "oven": "烤炉", 
        "toaster": "烤面包机", "sink": "下沉", "refrigerator": "冰箱", "book": "书籍", "clock": "时钟", "vase": "花瓶", "scissors": "剪刀", 
        "teddy bear": "泰迪熊", "hair drier": "干燥器", "toothbrush": "牙刷"
    },
    "jmjs": {
        "name": "静默监视",
        "sb_bx": "设备变形",
        "sb_dl": "设备断裂",
        "sb_qx": "设备倾斜"
    },
    "meter": {
        "name": "表记",
        "meter": "表记",
        "pointer": "指针"
    },
    "action_recognition":{
        "name": "行为识别",
        "smoking": "抽烟",
        "fall": "摔倒",
        "open_door": "开门",
        "fight": "打架"
    },
    "digital": {
        "name": "led数字",
        '0': '0',
        '1': '1',
        '2': '2',
        '3': '3',
        '4': '4',
        '5': '5',
        '6': '6',
        '7': '7',
        '8': '8',
        '9': '9',
        '0.': '0.',
        '1.': '1.',
        '2.': '2.',
        '3.': '3.',
        '4.': '4.',
        '5.': '5.',
        '6.': '6.',
        '7.': '7.',
        '8.': '8.',
        '9.': '9.',
        '-': '-'
    },
    "counter": {
        "name": "动作次数器",
        '0': '0',
        '1': '1',
        '2': '2',
        '3': '3',
        '4': '4',
        '5': '5',
        '6': '6',
        '7': '7',
        '8': '8',
        '9': '9'
    },
    "fire_smoke": {
        "name": "烟火",
        'fire': '火焰',
        'smoke': '烟火警告',

    },
    "helmet": {
        "name": "安全帽",
        'person': '未戴安全帽',
        'hat': '安全帽',
        'leifenghat': '棉帽',
    },
    "insulator": {
        "name": "绝缘子",
        "insulator": "绝缘子"
    },
    "sitecar": {
        "name": "机动车",
        'crane': '起重机',
        'excavator': '挖掘机'
    },
    "yingguangfu": {
        "name": "荧光服",
        'yingguangfu': '荧光服',
        'no_yingguangfu': '未穿工作服',
        'gongzuofu': '工作服'
    },
    "tools":{
        'name': '工具',
        'Ground wire': '接地线',
        'Electricity toolbox': '电工具箱',
        'DC electroscope': '直流验电器',
        'Electroscope': '验电笔',
        'Multimeter': '万用表',
        'Electric iron': '电熨头',
        'Tin cleaner': '吸锡器',
        'Solder wire': '焊锡线',
        'Electrical tape': '电胶带',
        'pliers': '钳子',
        'Network pliers': '网络钳',
        'Wrench': '扳手',
        'Small screwdriver': '小螺丝刀',
        'Screwdriver': '螺丝刀',
        'Hex screwdriver': '六角螺丝刀',
        'Hammer': '锤子',
        'Utility knife': '多功能刀',
        'Flashlight': '手电筒',
        'Tape measure': '卷尺'
    },
    "equipment_defects": {
        "name": "设备部件与缺陷",
        '变压器_本体': '变压器_本体',
        '变压器_套管': '变压器_套管',
        '变压器_冷却器': '变压器_冷却器',
        '变压器_冷却器_风扇': '变压器_冷却器_风扇',
        '变压器_冷却器_散热片': '变压器_冷却器_散热片',
        '变压器_呼吸器': '变压器_呼吸器',
        '变压器_油枕': '变压器_油枕',
        '变压器_机构箱': '变压器_机构箱',
        '变压器_端子箱': '变压器_端子箱',
        '断路器_本体': '断路器_本体',
        '断路器_操作箱': '断路器_操作箱',
        '刀闸_隔离刀闸': '刀闸_隔离刀闸',
        '刀闸_操作机构箱': '刀闸_操作机构箱',
        '刀闸_套管': '刀闸_套管',
        '表计': '表计',
        '绝缘子': '绝缘子',
        '部件表面油污': '部件表面油污',
        '地面油污': '地面油污',
        '金属锈蚀': '金属锈蚀',
        '表计破损': '表计破损',
        '部件外观异常': '部件外观异常',
        '呼吸器破损': '呼吸器破损',
        '箱门闭合异常': '箱门闭合异常',
        '绝缘子破裂': '绝缘子破裂',
        '异物': '异物',
        '缺陷不知道': '缺陷不知道',
        '设备不知道_操作机构箱': '设备不知道_操作机构箱'
    },
        "disconnector_texie": {
        "name": "刀闸状态-无模板配置",
        "he": "合闸正常",
        "fen": "分闸正常",
        "budaowei": "分合异常"
    },
    "level_gauge": {
        "name": "油位液位",
        "oil": "油位"}
}

defect_LIST = ['sly_bjbmyw','sly_dmyw','pzqcd','jyhbx','drqgd','jyz_pl','yxdgsg','jdyxxsd',
        'jsxs_ddjt','jsxs_ddyx','jsxs_jdyxx','jsxs_ecjxh','hzyw','bmwh','ws_ywyc','yljdq_flow',
        'yljdq_stop','fhz_ztyc','hxq_gjbs','ywzt_yfyc','hxq_gjtps','hxq_yfps','bj_wkps','bj_bpps',
        'bjdsyc_zz','bjdsyc_sx','bjdsyc_ywj','bjdsyc_ywc','bj_bpmh','gcc_mh','gcc_ps','xmbhyc',
        'yw_nc','yw_gkxfw','kgg_ybh','kk_h','zsd_l','mbhp','wcgz','wcaqm','xy']

AI_FUNCTION = {
    "表计读数": {
        "指针表计读数": ["指针式套管油位表", "油温表", "避雷器泄露电流表", "SF6压力表", "液压操动机构压力表", "分接档位表", "二次电流表"],
        "数字表计读数": ["避雷器动作次数表", "显示屏数显表", "led数字表"],
        "液位表计读数": ["油枕油位表", "液位式套管油位表"]
    },
    "刀闸状态识别": {
        "刀闸状态识别":["剪刀式刀闸", "摆臂式刀闸", "接地刀闸"]
    },
    "文本内容识别": {
        "二维码识别":["二维码识别"],
        "文本识别": ["标识牌内容识别", "车牌识别"]
    },
    "设备状态与缺陷检测": {
        "渗漏油": ["部件表面油污", "地面油污"],
        "设备破损变形": ["金属膨胀器冲顶破损", "均压环破损变形", "电容器鼓肚", "外绝缘裂纹破损", "引线断股松股", "接地引下线松动"],
        "金属锈蚀": ["导电接头", "导电引线", "接地引下线", "二次接线盒"],
        "设备烟火": ["设备烟火"],
        "设备积污": ["外绝缘表面污秽"],
        "状态指示": ["瓦斯观察窗油位下降", "油流继电器指示", "分合闸指示", "呼吸器硅胶变色", "油封油位异常", "硅胶筒破损", "呼吸器油封破损（含缺失）"],
        "表计读数异常": ["表计破损", "指针表表计读数异常", "数显表表计读数异常", "油位表计读数异常", "油位观察窗异常", "表计模糊", "观察窗模糊", "观察窗破损"],
        "箱门闭合异常": ["箱门闭合异常", "箱门闭合正常"],
        "异物": ["鸟巢", "挂空悬浮物"],
        "二次设备状态指示": ["压板状态", "空开状态", "指示灯", "面板显示异常"],
        "人员行为异常": ["未穿工装", "未穿安全帽", "吸烟"]
    },
    "其他": {
        "人员检测": ["人员检测"],
        "人员动作识别": ["摔倒", "开门", "打架"],
        "kk把手": ["kk把手方向识别"]
    },
}

def convert_ai_function(AI_FUNCTION):
    data = []
    count_1 = 0
    count_2 = len(AI_FUNCTION)
    count_3 = count_2
    for label_1 in AI_FUNCTION:
        count_3 += len(AI_FUNCTION[label_1])
    for label_1 in AI_FUNCTION:
        count_1 += 1
        data_1 = {}
        data_1["id"] = count_1
        data_1["label"] = label_1
        data_1["children"] = []
        for label_2 in AI_FUNCTION[label_1]:
            count_2 += 1
            data_2 = {}
            data_2["id"] = count_2
            data_2["label"] = label_2
            data_2["children"] = []
            for label_3 in AI_FUNCTION[label_1][label_2]:
                count_3 += 1
                data_3 = {}
                data_3["id"] = count_3
                data_3["label"] = label_3
                data_2["children"].append(data_3)
            data_1["children"].append(data_2)
        data.append(data_1)
    ai_function = {"data": data}
    return ai_function



## 不同颜色对应的hsv范围
## https://blog.csdn.net/qq_41895190/article/details/82791426
COLOR_HSV_MAP = {
    "black": [[0, 0, 0], [180, 255, 46]],
    "gray": [[0, 0, 46], [180, 43, 220]],
    "white": [[0, 0, 221], [180, 30, 255]],
    "red": [[156, 43, 46], [180, 255, 255]],
    "red2": [[0, 43, 46], [10, 255, 255]],
    "orange": [[11, 43, 46], [25, 255, 255]],
    "yellow": [[26, 43, 46], [34, 255, 255]],
    "green": [[35, 43, 46], [77, 255, 255]],
    "cyan": [[78, 43, 46], [99, 255, 255]], # 青色
    "blue": [[100, 43, 46], [124, 255, 255]],
    "purple": [[125, 43, 46], [155, 255, 255]] # 紫色
}

def convert_label(label, type_):
    for l in OBJECT_MAP[type_]:
        if label == OBJECT_MAP[type_][l]:
            return l
    if l == "bjdsyc":
        label = "bjdsyc_zz"
    if l == "yxdghsg":
        label = "yxdgsg"
    
    return label
