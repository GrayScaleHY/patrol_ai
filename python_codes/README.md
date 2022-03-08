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
应用代码一般以app_inspection开头，打包成可以用python直接调用的函数，输入输出对应接口设计的输入输出。注意，模型加载代码不要写进函数里，防止每次调用函数重新加载模型花费太多时间。
将新开发的应用添加到服务代码[util_inspection_server.py](https://git.utapp.cn/yuanhui/image/-/blob/main/python_codes/util_inspection_server.py)中。

