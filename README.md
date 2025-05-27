# 巡检算法服务-华为版本
该项目存放巡检图像算法所需的代码

### 巡检算法服务的部署
注意，部署服务器必须配备华为atlas加速卡。
##### 1. 准备部署文件
(1). 下载两个部署压缩包PatrolAi.zip和ascend_yolo_7.0.1.tar.gz。链接公司内网，将192.168.100.42加入到dns中，登录共享盘，内网链接为[https://seafile.utai.cn/d/3ce914e415ab4b1d89d4/](https://seafile.utai.cn/d/3ce914e415ab4b1d89d4/)。下载PatrolAi.zip和ascend_yolo_7.0.1.tar.gz两个部署包。  
(2). 服务器上新建文件夹/data, 将两个压缩包上传到/data目录下，终端输入```cd /data && unzip PatrolAi.zip```进行解压缩包。若出现如下所示文件结构，表示部署文件准备完毕。
```
  /data/
    PatrolAi/
    PatrolAi.zip
    ascend_yolo_7.0.1.tar.gz
```
##### 2. 安装华为加速卡驱动
终端输入```lspci | grep -i Device```命令查看服务器上是否正确具备硬件华为加速卡，若出现如下类似打印，则表示已具备硬件加速卡。
```
01:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d500 (rev 23)
06:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d500 (rev 23)
```
终端输入```npu-smi info```命令查看服务器是否正确安装了显卡驱动，若出现如下类似打印，表示显卡驱动安装完成。
```
+------------------------------------------------------------------------------+
| npu-smi 23.0.rc3                                 Version: 23.0.rc3           |
+--------------+-------------+-------------------------------------------------+
| NPU   Name   | Health      | Power(W)  Temp(C)           Hugepages-Usage(page|
| Chip  Device | Bus-Id      | AICore(%) Memory-Usage(MB)                      |
+============================+===========+====================================+
| 0     310P3  | OK          | NA        51                996   / 996         |
| 0     0      | 0000:01:00.0| 0         3841 / 21527                          |
+==============+=============+=================================================+
| 64    310P3  | OK          | NA        48                0     / 0           |
| 0     1      | 0000:06:00.0| 0         1764 / 21527                          |
+==============+=================+=============================================+
```
若未安装驱动，则执行以下命令完成安装
```
cd /data/PatrolAi/install/cann-hdk
chmod +x *
adduser HwHiAiUser
./Ascend-hdk-310p-npu-driver_23.0.1_linux-aarch64.run --full

```
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
##### 4. 加载巡检算法docker镜像
输入以下命令加载docker镜像，注意，输入命令后需要等待较长时间，请耐心等待。
```
cd /data
sudo docker load --input ascend_yolo_7.0.1.tar.gz
```
加载完成后，输入```sudo docker images```, 若出现以下docker镜像信息，表示docker加载成功。
```
REPOSITORY           TAG                                  IMAGE ID       CREATED          SIZE
utdnn/patrol_ai      ascend_yolo_7.0.1                    a63b0b8899ed   40 minutes ago   18GB
```
##### 5.启动巡检算法服务
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
