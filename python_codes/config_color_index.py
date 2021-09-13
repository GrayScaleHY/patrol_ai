import cv2


COLOR_MAP = {
    0: (0, 0, 0), # 黑色
    1: (255, 255, 255), # 白色
    2: (255, 0, 0), # 红色
    3: (0, 255, 0), # 绿色
    4: (0, 0, 255), # 蓝色
    5: (0, 255, 255), # 青色
    6: (255, 0, 255), # 洋红色
    7: (255, 255, 0), # 黄色

}


if __name__ == '__main__':
    import numpy as np
    a = np.ones([100, 100,3], dtype=int)
    b = a * [255, 255, 0]
    cv2.imshow("result", b)
    cv2.waitKey(1)
    print(b)

