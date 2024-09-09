import numpy as np

class GetInputData:
    """
    获取巡视输入信息。
    """

    def __init__(self, data):
        # self.input_data = input_data
        self.checkpoint = self.get_checkpoint(data)  # 点位名称
        self.config = self.get_config(data)  # 模板信息
        self.p, self.t, self.z = self.get_ptz(self.config)
        self.center, self.resize_rate, self.size = self.get_rectangle_info(self.config)
        self.fov_h, self.fov_v = self.get_fov(self.config)
        self.direction_p, self.direction_t = self.get_direction(self.config)
        self.range_p, self.range_t, self.range_z = self.get_range(self.config)
        
    def get_checkpoint(self, data):
        """
        获取checkpoint(巡检点位名称)。
        """
        if "checkpoint" in data and isinstance(data["checkpoint"], str):
            checkpoint = data["checkpoint"]
        else:
            checkpoint = ""
        return checkpoint
    
    def get_config(self, data):
        """
        获取模板信息。
        """
        if "config" in data and isinstance(data["config"], dict):
            config = data["config"]
        else:
            config = {}
        return config
    
    def get_ptz(self, config):
        p, t, z = config["ptz_coords"]
        return p, t, z
    
    def get_rectangle_info(self, config):
        w = config["Width"]
        h = config["Height"]
        b = config["rectangle_coords"]
        center = [int((w*b[2] + w*b[0])/2), int((h*b[3] + h*b[1])/2)]
        resize_rate = 1 / (max(b[2]-b[0], b[3]-b[1]))
        size = [w, h]
        return center, resize_rate, size
    
    def get_fov(self, config):
        fov_h = config["Horizontal"]
        fov_v = config["Vertical"]
        return fov_h, fov_v
    
    def get_direction(self, config):
        if "direction_p" in config and isinstance(config["direction_p"], int) and config["direction_p"] != -1:
            direction_p = int(config["direction_p"])
        else:
            direction_p = 1
        
        if "direction_t" in config and isinstance(config["direction_t"], int) and config["direction_t"] != -1:
            direction_t = int(config["direction_t"])
        else:
            direction_t = 1
        return direction_p, direction_t
    
    def get_range(self, config):
        if "range_p" in config and isinstance(config["range_p"], list) and len(config["range_p"]) > 0:
            range_p = config["range_p"]
        else:
            range_p = [0, 360]
        
        if "range_t" in config and isinstance(config["range_t"], list) and len(config["range_t"]) > 0:
            range_t = config["range_t"]
        else:
            range_t = [-5, 90]

        if "range_z" in config and isinstance(config["range_z"], list) and len(config["range_z"]) > 0:
                range_z = config["range_z"]
        else:
            range_z = [1, 25]
        
        return range_p, range_t, range_z

def adjust_camera(input_data):
    # 解析input_data
    DATA = GetInputData(input_data)
    checkpoint= DATA.checkpoint
    p = DATA.p
    t = DATA.t
    z = DATA.z
    x, y = DATA.center # 框子中心点坐标
    resize_rate = DATA.resize_rate
    W, H = DATA.size
    fov_h = DATA.fov_h
    fov_v = DATA.fov_v
    range_p = DATA.range_p
    range_t = DATA.range_t
    range_z = DATA.range_z
    direction_p = DATA.direction_p
    direction_t = DATA.direction_t

    # 将框子中心点坐标移到画面中心
    if x > (W / 2): # 水平方向
        delt_x = np.rad2deg(np.arctan((x - W / 2) / (W / 2) * np.tan( np.deg2rad(fov_h /2))))
    else:
        delt_x = -np.rad2deg(np.arctan((W / 2 - x) / (W / 2) * np.tan( np.deg2rad(fov_h /2))))
    
    if y > (H / 2): # 垂直方向
        delt_y = np.rad2deg(np.arctan((y - H / 2) / (H / 2) * np.tan( np.deg2rad(fov_v/ 2))))
    else:
        delt_y = -np.rad2deg(np.arctan((H / 2 - y) / (H / 2) * np.tan( np.deg2rad(fov_v/ 2))))
    
    if direction_p == 1:
        new_p = p + delt_x
    else:
        new_p = p - delt_x
    
    if direction_t == 1:
        new_t = t + delt_y
    else:
        new_t = t - delt_y
    
    # 根据p\t的范围调试得到的p\t值
    if new_p < range_p[0]:
        if range_p[0] == 0 and range_p[1] == 360:
            new_p = 360 + new_p
        else:
            new_p = range_p[0]
    if new_p > range_p[1]:
        if range_p[0] == 0 and range_p[1] == 360:
            new_p = new_p - 360
        else:
            new_p = range_p[1]
    
    if new_t < range_t[0]:
        new_t = range_t[0]
    
    if new_t > range_t[1]:
        new_t = range_t[1]
    
    # 计算new_z
    new_z = z * resize_rate
    if new_z < range_z[0]:
        new_z = range_z[0]
    if new_z > range_z[1]:
        new_z = range_z[1]
    
    out_data = {"code": 0, "data": {"ptz_new": [new_p, new_t, new_z]}}

    return out_data

if __name__ == "__main__":
    input_data = {
        "checkpoint": "预置位1", 
        "config":{
            "ptz_coords": [316.5, 4.099999904632568, 3.0], 
            "rectangle_coords": [0.9147727,0.121527778,0.94886364,0.17708333],
            "Horizontal": 20.899999618530273,
            "Vertical": 11.84000015258789,
            "Width": 704,
            "Height": 576,
            "direction_p": 1,
            "direction_t": 1,
            "range_p": [0, 360],
            "range_t": [-5, 90]
                }, 
        "type": "adjust_camera"
    }
    out_data = adjust_camera(input_data)
    print(out_data)