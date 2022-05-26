
## 目标检测中各标签的命名
OBJECT_MAP = {
    "meter": {
        "name": "表记",
        "meter": "表记",
        "pointer": "指针"
    },
    "air_switch": {
        "name": "空气开关",
        "air_switch_on": "合闸",
        "air_switch_off": "分闸"
        },
    "pressplate":{
        "name": "压板",
        "kgg_ybf": "退出",
        "kgg_ybh": "投入"
    },
    "led": {
        "name": "LED灯",
        "green_on": "绿灯亮",
        "green_off": "绿灯灭",
        "red_on": "红灯亮",
        "red_off": "红灯灭",
        "white_on": "白灯亮",
        "white_off": "白灯灭",
        "yellow_on": "黄灯亮",
        "yellow_off": "黄灯灭",
        "black_off": "黑灯灭",
    },
    "fanpaiqi": {
        "name": "翻拍器",
        "fanpaiqi_red": "翻牌器合",
        "fanpaiqi_green": "翻牌器分",
        "fanpaiqi_jiedi": "翻牌器接地"
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
    "rotary_switch":{
        'name': '旋钮开关',
        'up': '上',
        'leftup': '左上',
        'rightup': '右上',
        'left': '左'
    },
    "arrow": {
        'name': '箭头仪表',
        'up': '上',
        'down': '下',
        'leftup': '左上',
        'leftdown': '左下',
        'right': '右',
        'left': '左',
        'rightup': '右上',
        'rightdown': '右下'
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
    "ErCiSheBei":{
        'xmbhyc': '箱门异常',
        'xmbhzc': '箱门正常',
        'kgg_ybh': '合',
        'kgg_ybf': '分',
        'kqkg_hz': '合闸',
        'kqkg_fz': '分闸',
        'fpq_h': '翻牌器合',
        'fpq_f': '翻牌器分',
        'fpq_jd': '翻牌器接地',
        'xnkg_s': '上',
        'xnkg_zs': '左上',
        'xnkg_ys': '右上',
        'xnkg_z': '左',
        'zsd_lvdl': '绿灯亮',
        'zsd_lvdm': '绿灯灭',
        'zsd_hongdl': '红灯亮',
        'zsd_hongdm': '红灯灭',
        'zsd_baidl': '白灯亮',
        'zsd_baidm': '白灯灭',
        'zsd_huangdl': '黄灯亮',
        'zsd_huangdm': '黄灯灭',
        'zsd_heidm': '黑灯灭',
        'ys': '钥匙'
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
    "door":{
        "name": "箱门",
        "xmbhyc": "开",
        "xmbhzc": "闭"
    },
    "key":{
        "name": "钥匙",
        "key": "钥匙"
    },
    "robot":{
        'door_abnormal':"箱门异常",
        'door_normal':"箱门闭合",
        'suspension':"挂空悬浮物",
        'nest':"鸟巢",
        'no_helmet':"未带安全帽",
        'helmet':"安全帽",
        'smoking':"吸烟"

    },
    "rec_defect": {
        "name": "巡检17类缺陷",
        'bj_bpmh': '表盘模糊',
        'sly_dmyw': '地面油污',
        'xmbhyc': '箱门闭合异常',
        'yw_gkxfw': '挂空悬浮物',
        'yw_nc': '鸟巢',
        'wcgz': '未穿工装',
        'hxq_gjbs': '硅胶变色',
        'kgg_ybh': '压板合',
        'bjdsyc': '表计读数异常',
        'wcaqm': '未戴安全帽',
        'jyz_pl': '绝缘子破裂',
        'gbps': '盖板破损',
        'bj_bpps': '表盘破损',
        'xy': '吸烟',
        'ywzt_yfyc': '呼吸器油封油位异常',
        'bj_wkps': '外壳破损',
        'hxq_gjtps': '硅胶筒破损'
    }

}

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