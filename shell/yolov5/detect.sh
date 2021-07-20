## 测试yolov5的脚本

if [ $# != 2 ]; then
    echo "Usage: $0 <weights> <test_dir>"
    echo "output inference result"
    exit 1
fi

weights=$1
test_dir=$2

## egs:
# weights=/data/yh/handover/codes/yolov5-v4/VOCdevkit/led/saved_models/20210716/exp14/weights/best.pt
# test_dir=/data/yh/handover/codes/yolov5-v4/inference/led_test

out_dir=${test_dir}_result


python detect.py  --conf 0.4 \
    --img-size 640 \
    --weights $weights \
    --source $test_dir \
    --project $out_dir \
    --iou-thres 0.45 \
    --device 0