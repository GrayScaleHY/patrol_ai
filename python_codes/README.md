## 巡检算法开发流程

##### 1. 算法接口
需要接口文档给调用的人，方便查看。接口文档至少包含请求URL、post数据格式、返回数据格式。一般以wiki的形式编写，例如[接口文档](https://git.utapp.cn/xunshi-ai/json-http-interface/-/wikis/%E6%99%BA%E8%83%BD%E5%B7%A1%E6%A3%80-%E5%88%80%E9%97%B8%E5%88%86%E5%90%88%E7%8A%B6%E6%80%81%E5%88%A4%E6%96%AD-%E6%97%A0%E9%85%8D%E7%BD%AE)。  
(1). URL, 算法调用地址。  
(2). post数据，算法服务接收到的数据，一般是json字典形式，包含"image"、"config"、"type"三个元素。
```
{
    "image": "",  # 待分析图片，以base64字符串格式存储。
    "config":{},  # 模板信息，无法通过图像直接分析的预设信息
    "type": ""    # 待分析类型
}
```
(3). 返回数据，算法分析后返回给调用端的结果数据，一般是json字典形式，包含"code"、"data"、"msg"、"img_result"四个元素。
```
{
    "code":0, # 分析状态码，0表示正常，1表示异常。
    "data": {}, # 算法分析的结果数据
    "msg": "", # 分析过程中的一些log信息，用于debug。
    "img_result": "" # 将分析结果显示在图片上，base64格式
}

```
##### 2. 编写代码
(1). 根据算法接口的输入输出编写主函数，一般以函数文件名一般以app开头，参考[app_yeweiji.py](https://git.utapp.cn/yuanhui/patrol_ai/-/blob/main/python_codes/app_yeweiji.py)  
(2). 将编写的算法加入到flask服务中，参考[util_patrol_server.py](https://git.utapp.cn/yuanhui/patrol_ai/-/blob/main/python_codes/util_patrol_server.py)
##### 3. 编写代码注意事项
(1). 返回的"img_result"尽量信息详尽，有利于往后查找算法错误  
(2). 写好的函数尽量想办法多测试，等更新到现场再发现问题的话，改起来就会相对麻烦了。
##### 4. 合并代码
将代码推到主分支。
