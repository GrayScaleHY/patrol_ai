# 巡检算法服务
该项目存放巡检图像算法所需的代码

### 巡检算法服务的部署
服务区需要装备英伟达显卡，且系统为ubuntu，部署流程如下：
##### 1. 安装显卡驱动
服务器连接外网，终端输入以下命令。（如果没联网，请下载驱动安装包后安装）
```
sudo apt-get update
sudo apt-get install nvidia-driver-470
```
终端输入```nvidia-smi```命令，屏幕出现以下形式的打印时表示显卡驱动安装完成。
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.103.01   Driver Version: 470.103.01   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:02:00.0 Off |                  N/A |
| 25%   22C    P8    21W / 260W |   8103MiB / 11016MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```
##### 2. 安装docker
服务器连接外网，终端输入以下命令。（如果没联网，请下载驱动安装包后安装）
```
curl -sSL https://get.daocloud.io/docker | sh
```
终端输入```sudo docker images```命令，屏幕出现以下形式的打印时表示docker安装成功。
```
REPOSITORY   TAG                          IMAGE ID       CREATED       SIZE
```
##### 3. 将算法服务需要的代码和模型放入服务器
在根目录下新建/data文件夹，下载[inspection](http://192.168.69.36/d/c8340061061a41369159/)文件夹，将inspection文件夹放入/data目录下。
##### 4. 拉取并启动巡检docker
链接了公司内网的情况下，输入以下命令拉取巡检docker镜像。（镜像较大，请耐心等待拉去完成）
```

```
输入```sudo docker images```, 屏幕出现以下形式的打印时表示docker拉取成功。
```
REPOSITORY   TAG                          IMAGE ID       CREATED       SIZE
yh/dnn       ub18-cuda11.1-conda-trt7.2   de9fc58182f4   2 weeks ago   45.2GB
```
##### 5. 启动docker，并设置开机自启动算法服务
输入以下命令启动docker， 注意，--gpus的数量设置为服务器cpu核数的一半。
```
sudo docker run -it --gpus '"device=0"' --cpus="8." --name yh_inspection -p 5000:5000 --ipc=host -v /data/inspection:/data/inspection yh/dnn:ub18-cuda11.1-conda-trt7.2 
```
进入docker后，编辑~/.bashrc文件，使得启动docker时会自动开启服务。
```
vim ~/.bashrc 
## 将以下两行拷贝到.bashrc文件的末尾，并保存
cd /data/inspection/image/python_codes
/root/miniconda3/envs/tf24/bin/python util_inspection_server.py
```
同时按Ctrl+P+Q退出docker，输入以下命令，使得巡检docker会随开机自启动
```
sudo docker update --restart=always yh_inspection
```
##### 至此，巡检算法服务部署完必
