##将pt文件转换成saved model的脚本

if [ $# != 4 ]; then
    echo "Usage: $0 <pt_model> <cfg> <closs_num> <gpu>"
    echo "output contains models"
    exit 1
fi

pt_model=$1
cfg=$2
closs_num=$3
device=$4

## egs:
# pt_model=runs/fanpaiqi/weights/best.pt
# cfg=runs/fanpaiqi/yolov5l-fanpaiqi.yaml
# closs_num=3
iou_thres=0.45

tf=${pt_model:0:-3}.pb
optf=${pt_model:0:-3}_opti.pb
output_dir=${pt_model%/*}

## 将pt模型转换成pb模型
echo "CUDA_VISIBLE_DEVICES=$device python models/tf.py --cfg $cfg --weights $pt_model"
CUDA_VISIBLE_DEVICES=$device python models/tf.py --cfg $cfg --weights $pt_model

## 将pb模型转换成_optf.pb模型
echo "CUDA_VISIBLE_DEVICES=$device python models/tf_opti.py --tf $tf --optf $optf"
CUDA_VISIBLE_DEVICES=$device python models/tf_opti.py --tf $tf --optf $optf

## 将_optf.pb转换成saved model
echo "CUDA_VISIBLE_DEVICES=$device python models/pb2savedmodel.py --output_dir $output_dir --pb_dir $optf --class_num $closs_num --iou_thres $iou_thres"
CUDA_VISIBLE_DEVICES=$device python models/pb2savedmodel.py --output_dir $output_dir --pb_dir $optf --class_num $closs_num --iou_thres $iou_thres