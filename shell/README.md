### 巡检送检算法性能测试说明书
根据该教程使用南京电科院的算法平台完成缺陷识别、缺陷判别、一键顺控的测试输出。
##### 1. 进入算法管理平台
进入虚拟桌面，如gw-shanghai，虚拟桌面为windows系统。再在虚拟桌面中使用ssh远程连接到ubuntu机器上，如远程地址：10.144.239.162 。
##### 2. 挂载输入输出文件夹路径
(1). 图片\视频路径统一放置在/home/ubuntu/data/220_input/  
(2). 使用sudo su进入超级用户，并输入以下命令完成文件夹挂载。注：此处的“//10.144.239.59/share/shanghai/”以及“PICRESULT/shanghai”中的“shanghai”为示例，每个厂家挂载的目录对应为终端登录的省级名称。
```
cd /home/ubuntu/data
mount -t cifs //10.144.239.59/share/shanghai/ PICRESULT/shanghai -o 'username=share,password=Abc123!@#',uid=1000,gid=1000,iocharset=utf8,rw,dir_mode=0777,file_mode=0777
```
(3). 将事先准备两个部署压缩包PatrolAi.zip和ut-inspection-cuda10.1.tar.gz (若cuda版本为11.4，使用ut-inspection-cuda11.4.tar.gz)上传到/home/ubuntu/data目录下，压缩包[下载链接](http://192.168.69.36/d/b688a5bd4f5e4772a9bd/)。终端输入```cd /home/ubuntu/data && unzip PatrolAi.zip```进行解压缩包。终端输入```chmod +x PatrolAi/patrol_ai/shell/algo_perf/*```添加执行权限。
##### 3. 配置 powersky
(1). 通过执行powersky来调用识别以及判别的算法程序。首先找出powersky所在的文件夹，打开config.ini，配置文件中只需修改调用识别以及判别程序的脚本的路径，其他不做更改，内容如下。注，我们的脚本路径应该是filepath_01=/mnt/data/PatrolAi/patrol_ai/shell/algo_perf/{sb.sh, pb.sh, yjsk.sh}
```
[TcpServer]
Ip=10.144.239.59
Port=18900
[UdpServer]
Ip=127.0.0.1
Port=11111
[shell]
#sb
filepath_01=/mnt/data/XXX/sb.sh
#pb
filepath_02=/mnt/data/XXX/pb.sh
```
(2).根据上面内容中的filepath_01中的sb.sh中的路径找到该路径/mnt/data/XXX/，进入/mnt/data/XXX/sb.sh,打开sb.sh，内容如下：
```
#！/bin/sh
python /mnt/data/XXX/sb.py
```
(3). 进行 /mnt/data/XXX/sb.py，打开sb.py，找到output路径如output_folder='mnt/data/PICRESULT/shanghai/'。上述路径都为docker映射路径：如本地路径为/home/ubuntu/data，docker映射路径为/mnt/data。
注：执行脚本结尾要加发送结束报文的代码。代码如下
```
## shell
echo -en '\xcd\x32\xcd\x32\xcd\x32\x02\x00\x00\x00' | netcat -u 127.0.0.1 11111
## python
import socket
message = b"\xcd\x32\xcd\x32\xcd\x32\x02\x00\x00\x00"
addr = ('172.0.0.1', 11111)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(s.sendto(message, addr))
s.close()
```
##### 4. 进入docker环境
(1). 加载巡检算法docker镜像，输入以下命令加载docker镜像，注意，输入命令后需要等待较长时间，请耐心等待。
```
cd data
sudo docker load --input ut-inspection-cuda10.1.tar.gz
```
加载完成后，输入```sudo docker images```, 若出现以下docker镜像信息，表示docker加载成功。
```
REPOSITORY        TAG                                   IMAGE ID       CREATED        SIZE
utdnn/inspection  cuda10.1-patrolai-opencv-cuda         86c8a25fae43   4 days ago     37.1GB
```
(2). 执行以下命令进入docker环境
```
nvidia-docker run -it --runtime nvidia --name ut-PatrolAi -v /home/ubuntu/data:/mnt/data utdnn/inspection:cuda10.1-patrolai-opencv-cuda /bin/bash
```
##### 5. 运行powersky
(1). 在docker环境下用命令行在挂载的输出文件夹中创建一个文件，看在10.144.239.59的share/shanghai中有无刚创建的文件。
(2). 在docker环境下进入powersky所在的目录运行./powersky，如权限不够用sudo ./powersky

