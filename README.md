# 巡检算法服务
该项目存放巡检图像算法所需的代码

### 巡检算法服务的部署（ubuntu 18.04）
该部署教程以ubuntu 18.04以上版本为例，注意，部署服务器必须配备英伟达显卡。凝思系统上部署请参考教程[凝思操作系统部署巡检算法服务](https://git.utapp.cn/yuanhui/image/-/wikis/凝思操作系统部署巡检算法服务)。
##### 1. 准备部署文件
(1). 下载两个部署压缩包PatrolAi.zip和ut-inspection.tar.gz。内网下载链接为[ \\\192.168.105.36\Outgoing\巡检算法部署 ]，外网下载链接为[http://61.145.230.152:8775/巡检算法部署/](http://61.145.230.152:8775/巡检算法部署/)。
(2). 服务器上新建文件夹/data, 将两个压缩包上传到/data目录下，终端输入```cd /data && unzip PatrolAi.zip```进行解压缩包。若出现如下所示文件结构，表示部署文件准备完毕。
```
  /data/
    PatrolAi/
    PatrolAi.zip
    ut-inspection.tar.gz
```
##### 2. 安装显卡驱动 (若已安装，跳过)
快捷键ctrl+alt+f3进入命令行模式，依次进行如下操作完成显卡驱动安装。
```
## 关闭lightdm
sudo service gdm stop  

## 安装显卡驱动
cd /data/PatrolAi/install
sudo chmod 777 NVIDIA-Linux-x86_64-510.60.02.run
sudo ./NVIDIA-Linux-x86_64-510.60.02.run -no-opengl-files
```
安装完成后，重启服务器，终端输入```nvidia-smi```命令，屏幕出现以下打印时表示显卡驱动安装完成。
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
##### 3. 安装docker (若已安装，跳过)
输入以下命令安装docker
```
cd /data/PatrolAi/install/ubuntu18.04
sudo chmod +x install_docker.sh
sudo ./install_docker.sh
```
安装完成后，终端输入```sudo docker images```命令，屏幕出现以下形式的打印时表示docker安装成功。
```
REPOSITORY   TAG                          IMAGE ID       CREATED       SIZE
```
##### 4. 安装nvidia-docker
输入以下命令安装nvidia-docker
```
cd /data/PatrolAi/install/ubuntu18.04
sudo chmod +x install_nvidia_docker.sh
sudo ./install_nvidia_docker.sh
```
安装完成后，重启服务器，终端输入```sudo nvidia-docker images```命令，屏幕出现以下形式的打印时表示docker安装成功。
```
REPOSITORY   TAG                          IMAGE ID       CREATED       SIZE
```
##### 5. 加载巡检算法docker镜像
输入以下命令加载docker镜像，注意，输入命令后需要等待较长时间，请耐心等待。
```
cd /data
sudo docker load --input ut-inspection.tar.gz
```
加载完成后，输入```sudo docker images```, 若出现以下docker镜像信息，表示docker加载成功。
```
REPOSITORY           TAG                                   IMAGE ID       CREATED        SIZE
utdnn/inspectio      cuda11.4-conda-cuml-opencv-gtk        86c8a25fae43   11 days ago     37.1GB
```
##### 6.启动巡检算法服务
输入以下命令启动巡检算法服务。
```
cd /data/PatrolAi
sudo chmod 777 run_PatrolAi.sh
sudo docker run -d --gpus '"device=0"' --cpus="8." -e LANG=C.UTF-8 --shm-size 6g --name ut-PatrolAi --restart=always -p 5000:5000 --ipc=host -v /data/PatrolAi:/data/PatrolAi --entrypoint "/data/PatrolAi/run_PatrolAi.sh" utdnn/inspection:cuda11.4-conda-cuml-opencv-gtk
```
等待2分钟左右，输入```sudo docker logs ut-PatrolAi --tail 100```, 若出现如下打印，则表示算法部署成功。
```
 * Running on http://127.0.0.1:5000
 * Running on http://172.17.0.2:5000 (Press CTRL+C to quit)
```
