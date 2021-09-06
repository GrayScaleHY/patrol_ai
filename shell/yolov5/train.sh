## 训练yolov5的脚本


if [ $# != 4 ]; then
    echo "Usage: $0 <data_path> <batch_size> <epochs> <device>"
    echo "output contains infos of saved models"
    exit 1
fi

data_path=$1
batch_size=$2
epochs=$3
device=$4

## egs:
# data_path=../data/led
# batch_size=4
# epochs=300
# device=0

img_size=640

time=$(date "+%Y-%m-%d-%H-%M")
name=${data_path##*/}
cfg_data=$data_path/voc_$name.yaml  # 数据信息
cfg_model=$data_path/yolov5_model_$name.yaml # 模型结构
pre_weights=${data_path%/*}/weights/yolov5l.pt # 预训练模型
save_dir=$data_path/saved_models/$time  # 训练后模型信息保存路径

## 删除数据集的缓存文件
train_cache=$data_path/labels/train.cache
val_cache=$data_path/labels/val.cache
if [ -f $train_cache ];then
  rm $train_cache
fi
if [ -f $val_cache ];then
  rm $val_cache
fi

## 训练
python train.py --img-size $img_size \
        --multi-scale \
        --batch-size $batch_size \
        --epochs $epochs \
        --data $cfg_data \
        --cfg $cfg_model \
        --weights $pre_weights \
        --project $save_dir \
        --device $device
        
        