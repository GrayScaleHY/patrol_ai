#!/bin/bash

# 使用多docker多进程加速算法性能测试
# args: 
#   1: 算法类型，可选pb(判别)、qxsb(缺陷识别)、yjsk(一键顺控)
#   2: 待测试图片/视频的路径
#   3: 输出结果文件的路径
#   4: cuda版本，可选cuda10.1、cuda11.4

if [ $# != 4 ]; then
    echo "Usage: $0 <opt_type> <in_dir> <out_dir> <cuda version>"
    echo "num of args is wrong !"
    exit 1
fi

opt_type=$1
in_dir=$2
out_dir=$3
v_cuda=$4

echo "opt_type=$opt_type"
echo "in_dir=$in_dir"
echo "out_dir=$out_dir"
echo "cuda version=$v_cuda"

startTime_s=`date +%s`

project_dir=$(dirname $(dirname $(cd `dirname $0`; pwd)))

# 指定docker镜像
if [ $v_cuda == 'cuda11.4' ]; then
    docker_iso="utdnn/inspection:cuda11.4-conda-cuml-opencv-gtk"
elif [ $v_cuda == 'cuda10.1' ]; then
    docker_iso="utdnn/inspection:cuda10.1-patrolai-opencv-cuda "
else
    echo "Erro: cuda version is wrong !"
    exit 1
fi

# 运行性能测试docker镜像
echo "run $opt_type !"
nvidia-docker run -d --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=0 --cpus 2 --name ${opt_type}1 \
    -v ${project_dir}:/data/PatrolAi -v $in_dir:$in_dir -v $out_dir:$out_dir \
    --entrypoint "/data/PatrolAi/patrol_ai/shell/lib_patrol_performance.sh" \
    $docker_iso $opt_type $in_dir $out_dir $v_cuda "1/4"

nvidia-docker run -d --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=0 --cpus 2 --name ${opt_type}2 \
    -v ${project_dir}:/data/PatrolAi -v $in_dir:$in_dir -v $out_dir:$out_dir \
    --entrypoint "/data/PatrolAi/patrol_ai/shell/lib_patrol_performance.sh" \
    $docker_iso $opt_type $in_dir $out_dir $v_cuda "2/4"

nvidia-docker run -d --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=1 --cpus 2 --name ${opt_type}3 \
    -v ${project_dir}:/data/PatrolAi -v $in_dir:$in_dir -v $out_dir:$out_dir \
    --entrypoint "/data/PatrolAi/patrol_ai/shell/lib_patrol_performance.sh" \
    $docker_iso $opt_type $in_dir $out_dir $v_cuda "3/4"

nvidia-docker run -it --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=1 --cpus 2 --name ${opt_type}4 \
    -v ${project_dir}:/data/PatrolAi -v $in_dir:$in_dir -v $out_dir:$out_dir \
    --entrypoint "/data/PatrolAi/patrol_ai/shell/lib_patrol_performance.sh" \
    $docker_iso $opt_type $in_dir $out_dir $v_cuda "4/4"

# 根据out_dir中文件是否增加来判断测试是否完成
if [ $opt_type == 'yjsk' ]; then
    sleep_time=20
else
    sleep_time=5
fi

f_num=$(ls -l $out_dir |grep "^-"|wc -l)
sleep $sleep_time
while (($(ls -l $out_dir |grep "^-"|wc -l) > $f_num))
do
    f_num=$(ls -l $out_dir |grep "^-"|wc -l)
    echo "file num: $f_num"
    sleep $sleep_time
done

echo "all job is done !"
echo "file num: $f_num"

endTime_s=`date +%s`
sumTime=$[ $endTime_s - $startTime_s ]
echo "Spend Total:$sumTime seconds"
