# 巡检算法服务
该项目存放巡检图像算法所需的代码

### 巡检算法服务的部署
注意，部署服务器必须配备英伟达显卡。
##### 1. 准备部署文件
(1). 下载两个部署压缩包PatrolAi.zip和ut_patrol_ai.tar.gz。内网下载链接为[ \\\192.168.105.36\Outgoing\巡检算法部署 ]，外网下载链接为[http://61.145.230.152:8775/巡检算法部署/](http://61.145.230.152:8775/巡检算法部署/)。  
(2). 服务器上新建文件夹/data, 将两个压缩包上传到/data目录下，终端输入```cd /data && unzip PatrolAi.zip```进行解压缩包。若出现如下所示文件结构，表示部署文件准备完毕。
```
  /data/
    PatrolAi/
    PatrolAi.zip
    ut_patrol_ai.tar.gz
```
##### 2. 检查是否安装了显卡硬件和显卡驱动
终端输入```lspci | grep -i nvidia```命令查看服务器上是否正确具备硬件显卡，若出现如下类似打印，则表示已具备硬件显卡。
```
02:00.0 VGA compatible controller: NVIDIA Corporation GV102 (rev a1)
02:00.1 Audio device: NVIDIA Corporation Device 10f7 (rev a1)
02:00.2 USB controller: NVIDIA Corporation Device 1ad6 (rev a1)
02:00.3 Serial bus controller [0c80]: NVIDIA Corporation Device 1ad7 (rev a1)
```
终端输入```nvidia-smi```命令查看服务器是否正确安装了显卡驱动，若出现如下类似打印，表示显卡驱动安装完成。
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
注：若未安装显卡硬件或显卡驱动，请联系相关人员安装好再执行下面步骤。
##### 3. 安装docker (若已安装，跳过)
输入以下命令完成docker的安装。
```
cd /data/PatrolAi/install/docker-24.0.7
sudo chmod +x install_docker.sh
sudo ./install_docker.sh
```
安装完成后，终端输入```sudo docker images```命令，屏幕出现以下形式的打印时表示docker安装成功。
```
REPOSITORY   TAG                          IMAGE ID       CREATED       SIZE
```
##### 4. 安装nvidia-toolkit
输入以下命令安装nvidia-toolkit
```
cd /data/PatrolAi/install/docker-24.0.7
sudo chmod +x install_nvidia_toolkit.sh
sudo ./install_nvidia_toolkit.sh
```
##### 5. 加载巡检算法docker镜像
输入以下命令加载docker镜像，注意，输入命令后需要等待较长时间，请耐心等待。
```
cd /data
sudo docker load --input ut_patrol_ai.tar.gz
```
加载完成后，输入```sudo docker images```, 若出现以下docker镜像信息，表示docker加载成功。
```
REPOSITORY           TAG                                   IMAGE ID       CREATED        SIZE
 utdnn/patrol_ai     cuda11.6                86c8a25fae43   11 days ago     16GB
```
##### 6.启动巡检算法服务
输入以下命令启动巡检算法服务。
```
cd /data/PatrolAi
chmod +x run.sh
./run.sh
```
等待2分钟左右，输入```sudo docker logs ut-PatrolAi --tail 100```, 若出现如下打印，则表示算法部署成功。
```
 * Running on http://127.0.0.1:29528
 * Running on http://172.17.0.2:28528 (Press CTRL+C to quit)
```
