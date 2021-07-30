## 测试yolov5的脚本

if [ $# != 2 ]; then
    echo "Usage: $0 <weights> <test_dir>"
    echo "output inference result"
    exit 1
fi

weights=$1  # 模型文件
test_dir=$2 # 测试图片的文件夹

# # egs:
# weights=/data/yh/handover/codes/yolov5-v4/VOCdevkit/led/saved_models/20210716/exp14/weights/best.pt
# test_dir=/data/yh/handover/codes/yolov5-v4/inference/led_test_2

out_dir=${test_dir}_result
img_size=640 # 图片resize大小
iou_thres=0.45 # 阈值
device=0 # gpu


python detect.py  --conf 0.4 \
    --img-size $img_size \
    --weights $weights \
    --source $test_dir \
    --project $out_dir \
    --iou-thres $iou_thres \
    --device $device 
