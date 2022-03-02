import sys
import random
import sympy
import math
from config_object_name import COLOR_HSV_MAP
import cv2
import numpy as np


class Logger(object):
    """
    将控制端log保存下来的方法。
    demo:
        sys.stdout = Logger("log.txt")
    """
    def __init__(self,fileN ="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN,"a")
 
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

def color_list(c_size):
    """
    生成一个颜色列表。
    """
    color_map = {
        # 0: (0, 0, 0), # 黑色
        # 1: (255, 255, 255), # 白色
        0: (0, 0, 255), # 红色
        1: (0, 255, 0), # 绿色
        2: (255, 0, 0), # 蓝色
        3: (0, 255, 255), # 黄色
        4: (255, 0, 255), # 粉色
        5: (255, 255, 0), # 淡蓝
    }
    colors = []
    if c_size <= len(color_map):
        for i in range(c_size):
            colors.append(color_map[i])
    else:
        colors = [color_map[c] for c in color_map]
        for i in range(c_size - len(color_map)):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            colors.append(color)
    return colors

def oil_high(s_oil, s_round):
    """
    根据油面积和圆面积求油位的高度
    args:
        s_oil: 油面积
        s_round: 圆面积
    return:
        h: 油位高度
    """
    R = math.sqrt(s_round / math.pi) #根据圆面积求圆半径
    S = s_oil

    if S > s_round / 2: # 油位在圆心之上
        d=sympy.Symbol('H') # 圆心到油位的距离

        ## 大扇形面积加上等腰三角形面积等于油面积
        fx = (sympy.pi - sympy.acos(d/R))*R**2  + R * sympy.cos(sympy.asin(d/R)) * d - S
        d = sympy.nsolve(fx,d,0) # 赋值解方程
        h = d + R

    elif S < s_round / 2: # 油位在圆心之下
        d=sympy.Symbol('H')

        ## 小扇形面积减去等腰三角形面积等于油面积
        fx = sympy.acos(d/R)*R**2  - R * sympy.cos(sympy.asin(d/R)) * d - S
        d = sympy.nsolve(fx,d,0) # 赋值解方程
        h =  R - d

    else:
        h = R

    return h

def color_area(img, color_list=["black","white","red","red2","orange","yellow","green","cyan","blue","purple"]):
    """
    根据hsv颜色空间判断图片中各颜色的面积。
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_dict = {}
    for color in color_list:
        hsv_lower = np.array(COLOR_HSV_MAP[color][0])
        hsv_upper = np.array(COLOR_HSV_MAP[color][1])
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

        # color_sum = np.sum(mask)

        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary,None,iterations=2)
        cnts, hiera = cv2.findContours(binary.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        color_sum = 0
        for c in cnts:
            color_sum += cv2.contourArea(c)

        color_dict[color] = color_sum
    
    return color_dict


if __name__ == '__main__':
    import cv2
    import numpy as np
    c_list = color_list(10)
    for c in c_list:
        a = np.ones([1000, 1000,3], dtype=np.uint8)
        b = a * list(c)
        b = b.astype(np.uint8)
        cv2.imshow(str(c), b)
        cv2.waitKey(0)
