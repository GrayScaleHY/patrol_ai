### 巡检送检算法性能测试说明书
版本：1.1.1.070217_release
根据该教程完成缺陷识别、缺陷判别、一键顺控的测试输出。
##### 1. 准备部署文件
(1). 事先准备两个部署压缩包PatrolAi.zip和ut-inspection-cuda10.1.tar.gz (若cuda版本为11.4，使用ut-inspection-cuda11.4.tar.gz)。  
(2). 服务器上新建文件夹data, 将两个压缩包上传到data目录下，终端输入```cd data && unzip PatrolAi.zip```进行解压缩包。若出现如下所示文件结构，表示部署文件准备完毕。
```
  data/
    PatrolAi/
    PatrolAi.zip
    ut-inspection-cuda10.1.tar.gz
```
##### 2. 安装运行环境
(1). 安装显卡驱动(若已安装,跳过)。快捷键ctrl+alt+f3进入命令行模式，依次进行如下操作完成显卡驱动安装。
```
sudo service gdm stop  
cd data/PatrolAi/install
sudo chmod 777 NVIDIA-Linux-x86_64-510.60.02.run
sudo ./NVIDIA-Linux-x86_64-510.60.02.run -no-opengl-files
```
(2). 安装docker (若已安装，跳过)
输入以下命令安装docker。
```
# 若已连接外网，建议输入下面命令在线安装
curl -sSL https://get.daocloud.io/docker | sh

# 若没有外网，输入下面命令离线安装
cd data/PatrolAi/install/ubuntu18.04
sudo chmod +x install_docker.sh
sudo ./install_docker.sh
```
(3). 安装nvidia-docker(若已安装，跳过)
输入以下命令安装nvidia-docker。
```
cd data/PatrolAi/install/ubuntu18.04
sudo chmod +x install_nvidia_docker.sh
sudo ./install_nvidia_docker.sh
```
(4). 加载巡检算法docker镜像
输入以下命令加载docker镜像，注意，输入命令后需要等待较长时间，请耐心等待。
```
cd data
sudo docker load --input ut-inspection-cuda10.1.tar.gz
```
加载完成后，输入```sudo docker images```, 若出现以下docker镜像信息，表示docker加载成功。
```
REPOSITORY        TAG                                   IMAGE ID       CREATED        SIZE
utdnn/inspection  cuda10.1-patrolai-opencv-cuda         86c8a25fae43   4 days ago     37.1GB
```
##### 3. 运行测试脚本
运行以下脚本完成性能测试。其中，测试类别为 "pb\qxsb\yjsk" 之一，cuda版本为 "cuda10.1\cuda11.4"之一。
```
cd data/PatrolAi/patrol_ai/shell
chmod +x *patrol_performance.sh
./run_patrol_performance.sh <测试类别> <待测图片文件夹路径> <结果保存文件夹路径> <cuda版本>
```
