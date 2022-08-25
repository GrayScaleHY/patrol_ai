#!/bin/bash

# 执行算法性能测试的脚本
# args: 
#   1: 算法类型，可选pb(判别)、qxsb(缺陷识别)、yjsk(一键顺控)
#   2: 待测试图片/视频的路径
#   3: 输出结果文件的路径
#   4: 

if [ $# != 5 ]; then
    echo "Usage: $0 <opt_type> <in_dir> <out_dir> <cuda version> <data part>"
    echo "num of args is wrong !"
    exit 1
fi

opt_type=$1
in_dir=$2
out_dir=$3
v_cuda=$4
data_part=$5

echo "opt_type=$opt_type"
echo "in_dir=$in_dir"
echo "out_dir=$out_dir"
echo "cuda version=$v_cuda"
echo "data part=$data_part"

# 指定docker中的python路径
cd /data/PatrolAi/patrol_ai/python_codes
if [ $v_cuda == 'cuda11.4' ]; then
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/root/miniconda3/envs/rapids-22.04/lib/
    python_path=/root/miniconda3/envs/rapids-22.04/bin/python
elif [ $v_cuda == 'cuda10.1' ]; then
    python_path=/usr/bin/python3
else
    echo "Erro: cuda version is wrong !"
fi

# 运行python脚本
if [ $opt_type == 'pb' ]; then
    echo "run panbie opts !"
    $python_path util_panbie.py --source $in_dir --out_dir $out_dir --data_part $data_part
elif [ $opt_type == 'qxsb' ]; then
    echo "run qxsb opts !"
    $python_path util_qxsb.py --source $in_dir --out_dir $out_dir --data_part $data_part
elif [ $opt_type == 'yjsk' ]; then
    $python_path util_yjsk.py --source $in_dir --out_dir $out_dir --data_part $data_part
else
    echo "Erro: opt type is wrong !"
if