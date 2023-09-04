
import math
import numpy as np
import cv2

def segment2angle(base_coor, tar_coor):
    """
    输入线段的两个端点坐标（图像坐标系，y轴朝下），返回该线段斜率转换为0-360度。
    args:
        base_coor: 基本坐标点，(x1, y1)
        tar_coor: 目标点,(x2, y2)
    return:
        返回正常直角坐标系的角度，0-360度。
    """
    dx = tar_coor[0] - base_coor[0]
    dy = tar_coor[1] - base_coor[1]
    if dx == 0:
        if dy > 0:
            angle = 270
        else:
            angle = 90
    else:
        tan = dy / dx
        if tan > 0:
            if dx > 0:
                angle = 360 - (math.atan(tan) * 180 / math.pi)
            else:
                angle = 180 - (math.atan(tan) * 180 / math.pi)
        else:
            if dx > 0:
                angle = -(math.atan(tan) * 180 / math.pi)
            else:
                angle = 180 - (math.atan(tan) * 180 / math.pi)
    return angle


def angle_scale(config):
    """
    根据刻度与角度的关系，求出相差一度对应的刻度差。
    args:
        config: list, 角度与刻度的对应关系。格式如下
                [{"136.39":0.0, "90.26":4.5, "72.12":8.0}, ..]
    return:
        out_cfg: list, 最小刻度和角度，单位刻度。格式如下：
                [[141.7, -0.1, 0.018]]
    """
    out_cfg = []
    for scale_config in config:
        assert len(scale_config) >= 2, "刻度信息至少要有两个。"

        # 将config转换成浮点数字型，并置于array中。
        config_list = np.array([[0, 0]]*len(scale_config), dtype=float)
        count = 0
        for ang in scale_config:
            config_list[count][0] = float(ang)
            config_list[count][1] = float(scale_config[ang])
            count += 1

        # 找出最小刻度的行index
        min_index = np.where(config_list[:, 1] == min(config_list[:, 1]))[0][0]
        util_scale = 0
        for i, cfg in enumerate(config_list):
            if i != min_index:

                # 根据指针大刻度在顺时针方向的原则获取角度跨度。
                if cfg[0] < config_list[min_index][0]:
                    util_scale += (cfg[1] - config_list[min_index]
                                   [1]) / (config_list[min_index][0] - cfg[0])
                else:
                    util_scale += (cfg[1] - config_list[min_index][1]) / \
                        (360 - cfg[0] + config_list[min_index][0])

        util_scale = util_scale / (len(config_list) - 1)
        out_cfg.append([config_list[min_index][0],
                       config_list[min_index][1], util_scale])
        # print(str(scale_config) + " --> " +
        #       str([config_list[min_index][0], config_list[min_index][1], util_scale]))

    return out_cfg


def angle2sclae(cfg, ang):
    """
    根据角度计算刻度。
    args:
        cfg: 刻度盘属性：[最小刻度对应的角度， 最小刻度值， 单位刻度]
        ang: 需要计算的角度
    return:
        scale: 指针的刻度。
    """
    if ang < cfg[0]:
        scale = cfg[1] + (cfg[0] - ang) * cfg[2]
    else:
        scale = cfg[1] + (cfg[0] + 360 - ang) * cfg[2]
    return scale

def point_distance_line(point, segment):
    """
    点到直线的距离, 设直线函数为 Ax + By + C = 0 , 使用距离公式:
    A = y2 - y1
    B = x1 - x2
    C = x1 * (y1 - y2) + y1 * (x2 - x1)
    dis = abs(A * x0 + B * y0 + C) / sqrt(A**2 + B**2)
    args:
        point: 点，格式[x0, y0]
        sigment: 直线上的线段[x1, y1, x2, y2]
    return:
        distance: 点到直线的距离
    """
    segment = np.array(segment, dtype=float)
    p0 = np.array(point, dtype=float)

    p1 = segment[:2] ## 线段的第一个点
    p2 = segment[-2:] ## 线段的第二个点

    #对于两点坐标为同一点时,返回点与点的距离
    if (p1 == p2).all():
        return np.linalg.norm(p0 -p1)

    #计算直线的三个参数
    A = p2[1] - p1[1]
    B = p1[0] - p2[0]
    C = p1[0] * (p1[1] - p2[1]) + p1[1] * (p2[0] - p1[0])

    #根据点到直线的距离公式计算距离
    distance = np.abs(A * p0[0] + B * p0[1] + C) / (np.sqrt(A**2 + B**2))

    return distance

def disp(p0, p1):
    return ((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2) ** 0.5

def intersection_pointers(line, arc):
    # line = [834, 830, 769, 694]
    # arc = [893, 828, 710, 735, 780, 702]
    line = np.array(line, dtype=float) ## int转float
    arc = np.array(arc, dtype=float)
    # 根据线段到圆心的远近，确定坐标的先后
    if ((line[0]-arc[0])**2 + (line[1]-arc[1])**2) > ((line[2]-arc[0])**2 + (line[3]-arc[1])**2):
        line = [line[2], line[3], line[0], line[1]]

    segment = arc[2:]
    coor = intersection_segment(line, segment)
    if coor is None:
        return None
    
    p_start = line[:2]; p_end = line[2:]
    if disp(coor, p_start) < disp(coor, p_end):
        if disp(coor, p_end) > disp(p_start, p_end):
            return None
    return coor


def intersection_arc(line, arc):
    """
    计算射线line与圆弧arc的交点坐标，最多返回一个坐标值。
    args:
        line: [x1, y1, x2, y2]
        arc: [xc, yc, x1, y1, x2, y2], 注意:半径r取(x1,y1)到圆心的距离。
    return:
        (x1, y1): 交点坐标。也可能是None，表示直线与弧线无交点。
    # """
    # line = [834, 830, 769, 694] # [x1, y1, x2, y2]
    # arc = [1574, 529, 1478, 432, 1555, 421]# [xc, yc, x1, y1, x2, y2]
    line = np.array(line, dtype=float) ## int转float
    arc = np.array(arc, dtype=float)
    # 根据线段到圆心的远近，确定坐标的先后
    if ((line[0]-arc[0])**2 + (line[1]-arc[1])**2) > ((line[2]-arc[0])**2 + (line[3]-arc[1])**2):
        line = [line[2], line[3], line[0], line[1]]

    ## 定义直线和圆形方程式
    lx1 = line[0]; ly1 = line[1]; lx2 = line[2]; ly2 = line[3]
    xc = arc[0]; yc = arc[1]
    ax1 = arc[2]; ay1 = arc[3]; ax2 = arc[4]; ay2 = arc[5]
    # 直线: y = a * x + b
    if lx1 == lx2:
        a = None
    else:
        a = (ly1 - ly2) / (lx1 - lx2)
        b = (ly1 - a * lx1)
    # 圆形: (x - xc)^2 + (y - yc)^2 = r2
    r2 = (yc - ay1) ** 2 + (xc - ax1) ** 2

    ## 判断直线与圆是否相交。
    dis = point_distance_line((xc, yc), line)
    if dis ** 2 > r2:
        return None

    ## 计算直线与圆弧的交点坐标
    if a is None:
        ## 直线斜率不存在，线段垂直与x轴
        x1 = lx1
        x2 = lx1
        y1 = math.sqrt(r2 - (x1 - xc) ** 2) - yc
        y2 = -math.sqrt(r2 - (x1 - xc) ** 2) - yc
    else:
        ## 若直线line经过圆弧arc任意端点，则，返回该端点。
        if int(ax1 * a + b) == int(ay1) and (lx1-ax1)**2+(ly1-ay1)**2 > (lx2-ax1)**2+(ly2-ay1)**2:
            return int(ax1), int(ay1)
        if int(ax2 * a + b) == int(ay2) and (lx1-ax2)**2+(ly1-ay2)**2 > (lx2-ax2)**2+(ly2-ay2)**2:
            return int(ax2), int(ay2)
        
        ## 有斜率的情况下，联立直线和圆形方程组得: Ax^2 + Bx + C = 0
        A = 1 + a ** 2
        B = 2 * a * (b - yc) - 2 * xc
        C = xc ** 2 + (b - yc) ** 2 - r2
        detl = B ** 2 - 4 * A * C
        if detl <= 0: # 相切或不相交
            return None
        else:
            ## 二元一次方程组求解公式
            x1 = (-B - math.sqrt(detl)) / (2 * A)
            x2 = (-B + math.sqrt(detl)) / (2 * A)
            y1 = a * x1 + b
            y2 = a * x2 + b
    
    ## 取离线段末端最近的交点坐标作为唯一的交点坐标。
    if ((lx2-x1)**2+(ly2-y1)**2) <= ((lx2-x2)**2+(ly2-y2)**2):
        x = x1; y = y1
    else:
        x = x2; y = y2

    ## 判断交点坐标是否在圆弧内
    ang1 = segment2angle((xc, yc), (ax1, ay1))
    ang2 = segment2angle((xc, yc), (x, y))
    ang3 = segment2angle((xc, yc), (ax2, ay2))
    if ang2 <= ang1:
        ang2_ = ang1 - ang2
    else:
        ang2_ = ang1 + (360 - ang2) # 计算圆弧arc的夹角度数
    if ang3 <= ang1:
        ang3_ = ang1 - ang3
    else:
        ang3_ = ang1 + (360 - ang3) # 计算圆弧arc的夹角度数
    if ang2_ > ang3_:
        return None
    
    return int(x), int(y)


def intersection_segment(line, segment):
    """
    求直线穿过线段的交点坐标。
    agrs:
        line: 在直线上的两点坐标[x0, y0, x1, y1]
        segment: 线段两端点的坐标[x0, y0, x1, y1]
    return:
        (x, y): 交点坐标（如果没交点，返回None）
    """
    ## 如果线段或直线只包含一个坐标点，则报错
    assert (line[0], line[1]) != (line[2], line[3]
                                  ), "line needs at least two different points"
    assert (segment[0], segment[1]) != (segment[2], segment[3]
                                        ), "segment needs at least two different points"

    ## 将坐标值转为浮点型，方便计算
    line = np.array(line, dtype=float)
    segment = np.array(segment, dtype=float)
    
    ## 计算直线的斜率k_l和偏执值b_l
    if line[2] !=  line[0]:
        k_l = (line[3] - line[1]) / (line[2] - line[0]) # k = (y2 - y1) / (x2 - x1)
        b_l = line[1] - line[0] * k_l  # b = y1 - x1 * k
    else:
        k_l = None
    ## 计算线段的斜率k_s和偏执值b_s
    if segment[2] !=  segment[0]: # 判断直线是否有斜率
        k_s = (segment[3] - segment[1]) / (segment[2] - segment[0]) # k = (y2 - y1) / (x2 - x1)
        b_s = segment[1] - segment[0] * k_s  # b = y1 - x1 * k
    else:
        k_s = None

    if k_l == k_s: ## 如果直线和线段的斜率相同，则返回None
        return None

    if k_l is not None:
        ## 根据线段两端点是否在直线两边，判断直线与线段是否有交点。
        ds_y1 = segment[1] - (k_l * segment[0] + b_l)
        ds_y2 = segment[3] - (k_l * segment[2] + b_l)
        if ds_y1 * ds_y2 > 0:
            return None
        
        if k_s is None:
            x = segment[0]
        else:
            x = (b_s - b_l) / (k_l - k_s) #联立方程式计算x
        y = k_l * x + b_l
    
    else:
        ##根据线段两端点是否再直线两边，判断直线与线段是否有交点。
        if (segment[0] - line[0]) * (segment[2] - line[0]) > 0:
            return None
        
        x = line[0]
        y = k_s * x + b_s
    
    return int(x), int(y)

def cfgs2segs(cfgs):
    """
    将分割模型输出拟合成线段
    args:
        cfgs: 分割模型的推理结果。
    return:
        cfgs: 增加"seg"元素，seg是mask拟合成的线段。
    """
    segs = []
    for i in range(len(cfgs)):
        cfg = cfgs[i]
        box = list(cfg["coor"])

        ## 查找mask中1元素的位置。
        mask = cfg["mask"]
        w = np.where(mask==1)
        pointers = np.array([[w[1][i], w[0][i]] for i in range(len(w[0]))])
        if len(pointers) == 0:
            cfgs[i]["seg"] = [0,0,10,10]
            continue

        ## 拟合直线
        fit_line = cv2.fitLine(pointers, cv2.DIST_L2, 0, 0.01, 0.01)
        fit_line = np.squeeze(fit_line)  # 去除冗余的维度
        cos_l = fit_line[0]; sin_l = fit_line[1]
        x_l = fit_line[2]; y_l = fit_line[3]

        ## 若拟合直线是垂直的，则y为y_l, x为box上下两边界。
        if -0.01 < sin_l < 0.01:
            seg = [int(box[0]), int((box[1]+box[3])/2), int(box[2]), int((box[1]+box[3])/2)]
        
        # 若拟合直线是水平的，则x为x_l, y_为box左右两边界。
        elif -0.01 < cos_l < 0.01:
            seg = [int((box[0]+box[2])/2), int(box[1]), int((box[0]+box[2])/2), int(box[3])]
        
        else:
            # 非水平或垂直的情况下
            tan_l = sin_l / cos_l 
            det_l = y_l - (tan_l * x_l) # 直线公式改为y = tan_l * x + det_l
            seg = []
            for x_p in [box[0], box[2]]: # 求与box上下两边界的交点
                y_p = tan_l * x_p + det_l
                if box[1] < y_p < box[3]: # 交点坐标的y值需要再box左右边界之间
                    seg = seg + [int(x_p), int(y_p)]
            
            for y_p in [box[1], box[3]]: # 求与box左右两边界的交点
                x_p = (y_p - det_l) / tan_l
                if box[0] <= x_p <= box[2]: # 交点左边的x值需要在box上下边界之间
                    seg = seg + [int(x_p), int(y_p)]
        
        if len(seg) == 4:
            cfgs[i]["seg"] = seg
        else:
            cfgs[i]["seg"] = [0,0,10,10]
    
    return cfgs

def contour2segment(contours, boxes):
    """
    将轮廓拟合成线段。
    args:
        contours: 轮廓的坐标集, [array[x, 1, 2], ..]
        img: image data
    return:
        segments: 线段，结构：[[x0, y0, x1, y1], ..]
    """
    assert len(contours) == len(boxes), "contours not match boxes!"
    segments = []
    for i, contour in enumerate(contours):
        ## 轮廓拟合成直线，返回[cosΘ, sinΘ, x0, y0]
        fit_line = cv2.fitLine(contour, cv2.DIST_HUBER, 0, 0.01, 0.01)
        fit_line = np.squeeze(fit_line)  # 去除冗余的维度
        cos_l = fit_line[0]
        sin_l = fit_line[1]
        x_l = fit_line[2]
        y_l = fit_line[3]

        ## 求能框住轮廓的最小矩形框的四个顶点坐标。
        # rect = cv2.minAreaRect(contour)
        # box = cv2.boxPoints(rect).astype(np.int64)  # 获取矩形四个定点坐标

        ## 将拟合的直线转换成[x1, y1, x2, y2]的形式
        if cos_l != 0: #是否有斜率
            tan_l = sin_l / cos_l
            line = [x_l, y_l, x_l + 1, y_l + tan_l * 1]
        else:
            line = [x_l, y_l, x_l, y_l + 1]

        ## 将box写成四条线段的形式
        box = list(boxes[i])
        seg_list = [[box[0], box[1], box[2], box[1]],
                    [box[2], box[1], box[2], box[3]],
                    [box[2], box[3], box[0], box[3]],
                    [box[0], box[3],box[0], box[1]]]
        
        ## 求直线与矩形框的交点坐标，理论上会有两个交点。
        coors = []
        segment = []
        for seg in seg_list:
            coor = intersection_segment(line, seg) # 求直线与线段的交点坐标
            if coor is not None and coor not in coors:
                coors.append(coor)
                segment = segment + list(coor)
        
        if len(segment) == 4:
            segments.append(segment)

    return segments

if __name__ == '__main__':
    coor = intersection_pointers(1, 2)
    print("val")
