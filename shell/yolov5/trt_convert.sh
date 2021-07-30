

weights=/data/yh/handover/codes/yolov5-v4/VOCdevkit/led/saved_models/2021-07-19-19-54/exp/weights/best.pt
device=cpu

onnx_file=${weights:0:-3}.onnx
out_file=${weights%/*}/model.plan
iou_thres=0.45
batch_size=8

## 生成onnx模型
echo "BOX_SCORE=True python export.py --weights $weights --opset 11"
BOX_SCORE=True python export.py --weights $weights --opset 11

## 生成trt模型
echo """
python trt_convert.py --onnx $onnx_file \
        -o $out_file \
        --batch_size $batch_size \
        --iou_thres $iou_thres \
        --ws-digits 30 \
        --explicit-batch --fp16 [--fp16] \
"""
python trt_convert.py --onnx $onnx_file \
        -o $out_file \
        --batch_size $batch_size \
        --iou_thres $iou_thres \
        --ws-digits 30 \
        --explicit-batch --fp16 --fp16 \