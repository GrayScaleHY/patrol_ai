import sys
import random


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
        0: (255, 0, 0), # 红色
        1: (0, 255, 0), # 绿色
        2: (0, 0, 255), # 蓝色
        3: (0, 255, 255), # 青色
        4: (255, 0, 255), # 洋红色
        5: (255, 255, 0), # 黄色
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
