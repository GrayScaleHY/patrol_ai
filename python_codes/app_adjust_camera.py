import numpy as np

class GetInputData:
    """
    获取巡视输入信息。
    """

    def __init__(self, data):
        # self.input_data = input_data
        self.checkpoint = self.get_checkpoint(data)  # 点位名称
        self.config = self.get_config(data)  # 模板信息
        self.range_p, self.range_t, self.range_z = self.get_range(self.config)
        self.p, self.t, self.z = self.get_ptz(self.config, self.range_p, self.range_t)
        self.center, self.resize_rate = self.get_rectangle_info(self.config)
        self.fov_h, self.fov_v = self.get_fov(self.config)
        self.direction_p, self.direction_t = self.get_direction(self.config)
        
        
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
    
    def get_ptz(self, config, range_p, range_t):
        p, t, z = config["ptz_coords"]
        if p > range_p[1]:
            p = p - 360
        
        if t > range_t[1]:
            t = t - 360
        return p, t, z
    
    def get_rectangle_info(self, config):
        b = config["rectangle_coords"]
        center = [(b[2] + b[0])/2, (b[3] + b[1])/2]
        resize_rate = 0.7 / (max(b[2]-b[0], b[3]-b[1]))
        return center, resize_rate
    
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
            range_t = [0, 90]

        if "range_z" in config and isinstance(config["range_z"], list) and len(config["range_z"]) > 0:
            range_z = config["range_z"]
        else:
            range_z = [1, 25]
        
        return range_p, range_t, range_z

def convert_pt(x, range_x):
    xmax = range_x[1]
    if range_x[0] < range_x[1]:
        xmin = range_x[0]
    else:
        xmin = range_x[0] - 360
    
    if range_x[0] == 0 and range_x[1] == 360:
        if x < 0:
            x = 360 + x

    if x > xmax:
        return xmax
    
    if x < xmin:
        x = xmin
    
    if x < 0:
        x = x + 360
    
    return x


def adjust_camera(input_data):
    # 解析input_data
    DATA = GetInputData(input_data)
    checkpoint= DATA.checkpoint
    p = DATA.p
    t = DATA.t
    z = DATA.z
    x, y = DATA.center # 框子中心点坐标
    resize_rate = DATA.resize_rate
    fov_h = DATA.fov_h
    fov_v = DATA.fov_v
    range_p = DATA.range_p
    range_t = DATA.range_t
    range_z = DATA.range_z
    direction_p = DATA.direction_p
    direction_t = DATA.direction_t

    # 将框子中心点坐标移到画面中心
    if x > 0.5: # 水平方向
        delt_x = np.rad2deg(np.arctan((x - 0.5) / 0.5 * np.tan( np.deg2rad(fov_h /2))))
    else:
        delt_x = -np.rad2deg(np.arctan((0.5 - x) / 0.5 * np.tan( np.deg2rad(fov_h /2))))
    
    if y > 0.5: # 垂直方向
        delt_y = np.rad2deg(np.arctan((y - 0.5) / 0.5 * np.tan( np.deg2rad(fov_v/ 2))))
    else:
        delt_y = -np.rad2deg(np.arctan((0.5 - y) / 0.5 * np.tan( np.deg2rad(fov_v/ 2))))
    
    print("delt_x:", delt_x)
    print("delt_y:", delt_y)

    if direction_p == 1:
        new_p = p + delt_x
    else:
        new_p = p - delt_x
    
    if direction_t == 1:
        new_t = t + delt_y
    else:
        new_t = t - delt_y

    # pt特殊转换
    new_p = convert_pt(new_p, range_p)
    new_t = convert_pt(new_t, range_t)
    
    # 计算new_z
    new_z = z * resize_rate
    if new_z < range_z[0]:
        new_z = range_z[0]
    if new_z > range_z[1]:
        new_z = range_z[1]
    
    out_data = {"code": 0, "data": {"ptz_new": [new_p, new_t, new_z]}}

    return out_data

if __name__ == "__main__":
    import requests

    ## 获取摄像机参数
    get_url = "http://192.168.44.143:31010/api/v1/channel/getGisInfo" # ?chnid=1669
    data = {"chnid": 5334}
    cam_info = requests.get(get_url, params=data).json()
    P = cam_info["data"]["PanPos"]
    T = cam_info["data"]["TiltPos"]
    Z = cam_info["data"]["ZoomPos"]
    FOV_H = cam_info["data"]["Horizontal"]
    FOV_V = cam_info["data"]["Vertical"]
    print("FOV_H:", FOV_H, "\tFOV_V:", FOV_V)
    print("P:", P, "\tT:", T, "\tZ:",Z)

    input_data = {
        "checkpoint": "预置位1", 
        "config":{
            "ptz_coords": [P, T, Z],
            "rectangle_coords": [0.2301166489925769, 0.7460035523978685, 0.264050901378579, 0.8063943161634103],
            "Horizontal": FOV_H,
            "Vertical": FOV_V,
            "direction_p": 2,
            "direction_t": 2,
            "range_p": [0, 360],
            "range_t": [0, 90],
            "range_z": [1,23]
                }, 
        "type": "adjust_camera"
    }
    out_data = adjust_camera(input_data)

    ptz = out_data["data"]["ptz_new"]
    data = {"z": ptz[2], "p": ptz[0], "t": ptz[1], "chnid": 5334}
    put_url = "http://192.168.44.143:31010/api/v1/channel/ptzPos"
    cam_put = requests.put(put_url, params=data).json()

    print("p:", ptz[0], "\tt:", ptz[1], "\tz", ptz[2])