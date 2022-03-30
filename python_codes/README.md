# 巡检算法系统和图像处理
该项目提供各种图像分析算法和巡检接口服务。

## 巡检算法开发流程

##### 设计接口
需要提供给机器人后端调用的的URL、输入格式、输出格式，以wiki的形式写[接口文档](https://git.utapp.cn/xunshi-ai/json-http-interface)。

输入格式如下：
```
{
    "image": "",  # 待分析图片，以base64字符串格式存储。
    "config":{},  # 模板信息，无法通过图像直接分析的预设信息
    "type": ""    # 待分析类型
}
```
输出格式如下：
```
{
    "code":0, # 请求成功与否
    "data": {}, # 算法分析的结果
    "msg": "", # 分析过程中的一些log信息，用于debug。
    "img_result": "" # 将分析结果显示在图片上，base64格式
}

```
##### 编写应用和服务代码
训练好的模型统一放到"/data/inspection"目录下，以方便更新和管理。各种lebel对应的中文名列举在[config_object_name.py](https://git.utapp.cn/utiva/image/-/blob/main/python_codes/config_object_name.py)下。应用代码一般以app_inspection开头，打包成可以用python直接调用的函数，输入输出对应接口设计的输入输出（注意，模型加载代码不要写进函数里，防止每次调用函数重新加载模型花费太多时间）。将新开发的应用添加到服务代码[util_inspection_server.py](https://git.utapp.cn/yuanhui/image/-/blob/main/python_codes/util_inspection_server.py)中。
##### 测试接口
1. 开启巡检服务，首先clone image项目，再在image目录下clone yolov5项目，然后开启docker，进入python环境，最后运行[util_inspection_server.py](https://git.utapp.cn/yuanhui/image/-/blob/main/python_codes/util_inspection_server.py)脚本。操作demo如下：
```
## 克隆本项目
git clone git@git.utapp.cn:yuanhui/image.git

## 在image目录下克隆yolov5项目，因为lib_inference_yolov5.py脚本需要调用yolov5项目的包
cd image
git clone https://github.com/ultralytics/yolov5.git

## 开启配置好巡检环境docker，注意-p要与util_inspection_server.py代码中的端口对应。
docker run -it --gpus '"device=1"' --cpus="8" --name yh_inspection -p 5000:5000 --ipc=host -v /home/yh/image:/home/yh/image -v /data/inspection:/data/inspection yh/dnn:ub18-cuda11.1-conda-trt7.2 

## 在docker中运行巡检系统python脚本
cd /home/yh/image/python_codes 
conda activate tf24 
python util_inspection_server.py 

## 设置进入docker自启动服务，将以下命令加入到~/.bashrc
cd /data/inspection/image/python_codes
/root/miniconda3/envs/tf24/bin/python util_inspection_server.py
```
2. 自己先使用[util_inspection_request.py](https://git.utapp.cn/utiva/image/-/blob/main/python_codes/util_inspection_request.py)代码是否有问题，若没问题,则会在"inspection_result"文件加下形成带结果显示的图片。
3. 将接口文档链接发送给张瑞广，让他使用巡检机器人测试下性能。

