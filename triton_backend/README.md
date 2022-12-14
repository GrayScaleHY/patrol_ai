# Triton python backend examples
该项目提供了mask-rcnn和CenterNet2的triton python backend示例，分别是"pointer"和"insulator"例子。

## triton的使用
参考链接[triton Quickstart](https://github.com/triton-inference-server/server/blob/main/docs/quickstart.md)

##### 开启服务 ( 以yolov5的tensorrt模型为例 )
准备好转换过的tensorrt模型，并且按以下目录结构存放。
```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.plan
```
运行triton docker，并且使用tritonserver命令开启服务。注意, tensorrt模型的triton服务需要匹配正确的gpu型号。
```
$ docker run --gpus=1 --rm --shm-size 4g -p8010:8000 -p8011:8001 -p8012:8002 -v /home/yh/triton_models/trt2080ti:/models nvcr.io/nvidia/tritonserver:20.12-py3 tritonserver --model-repository=/models
```
当服务端出现以下信息时表示服务以成功开启，且可以使用GRPC, HTTP, Metrics协议来请求服务
```
I1002 21:58:57.891440 62 grpc_server.cc:3914] Started GRPCInferenceService at 0.0.0.0:8001
I1002 21:58:57.893177 62 http_server.cc:2717] Started HTTPService at 0.0.0.0:8000
I1002 21:58:57.935518 62 http_server.cc:2736] Started Metrics Service at 0.0.0.0:8002
```
TensorRT、saved-model、torchscript和ONNX模型在triton上支持度比较高，用server内部的C++逻辑就能跑。有些甚至不需要模型配置文件，Triton可以自动获得所有必需的设置。
##### 客户端请求
查看triton服务的输入输出
```
$ curl -v 192.168.57.159:8010/v2/models/airswitch | jq
```
有两种客户端方式，1.[官方提供客户端](https://github.com/triton-inference-server/client) 2.自己使用requests.post函数写一个。

我们智能分析上triton服务对应的triton python [http客户端](https://git.utapp.cn/fangjiacong/yolov5/-/blob/add_http_infer/trt_client_image_v2.py) 和 [grpc客户端](https://git.utapp.cn/fangjiacong/yolov5/-/blob/add_http_infer/triton_client_grpc.py)。内有详细注释。
```
python trt_client_image_v2.py --input /home/yh/triton_models/airswitch.jpg --out /home/yh/triton_models/airswitch_result.jpg --url 192.168.57.159:8010 --model airswitch --model_type tensorrt
```
## 自定义triton python后端
参考[官方教程](https://github.com/triton-inference-server/python_backend)和[detectron2例子](https://github.com/triton-inference-server/server/issues/3074)。

1. 创建triton基础环境。可以参考[教程](https://github.com/triton-inference-server/python_backend)在本地build一个，也可以直接使用提供的triton环境docker。

2. 在triton环境的基础上搭建推理所需要的环境。例如，detectron2可以用以下命令搭建。
```
$ apt-get update && apt-get install -y python3-dev
$ python3 -m pip install --upgrade pip
$ python3 -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
$ python3 -m pip install opencv-contrib-python-headless
$ git clone https://github.com/facebookresearch/detectron2.git
$ cd detectron2   
$ git checkout v0.5 ## 若要跑CenterNet2,则需要使用detectron2 v0.5版本
$ python3 setup.py build develop
```
3. 参考[model.py](https://github.com/triton-inference-server/python_backend/blob/main/examples/add_sub/model.py)写运行文件，构建TritonPythonModel类和initialize，execute和finalize这三个函数。model.py和config.pbtxt按照以下目录结构存放。例如上面的两个例子,"pointer"是mask-rcnn指针模型，"insulator"是CenterNet2绝缘子模型。
```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.py
```
4. 运行服务。(注意，模型较多或者当个模型较大时要加上--shm-size参数)
```
$ docker run --gpus=1 --rm --shm-size 4g -p8010:8000 -p8011:8001 -p8012:8002 -v /home/yh/triton_models/python_backend:/models ut/tritonserver:detectron2-v0.5 tritonserver --model-repository=/models
```
5. 运行客户端测试
```
python trt_client_image_v2.py --input /home/yh/triton_models/pointer.jpg --out /home/yh/triton_models/pointer_result.jpg --url 192.168.57.159:8010 --model pointer --model_type tensorrt
python trt_client_image_v2.py --input /home/yh/triton_models/insulator.jpg --out /home/yh/triton_models/insulator_result.jpg --url 192.168.57.159:8010 --model insulator --model_type tensorrt
```





