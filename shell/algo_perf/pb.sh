in_dir=/mnt/data/220_input/pb
out_dir=/mnt/data/PICRESULT/shanghai/pb_result

startTime_s=`date +%s`

cd /mnt/data/PatrolAi/patrol_ai/python_codes
CUDA_VISIBLE_DEVICES=0 python3 util_panbie.py --source $in_dir --out_dir $out_dir --data_part 1/3 &
CUDA_VISIBLE_DEVICES=0 python3 util_panbie.py --source $in_dir --out_dir $out_dir --data_part 2/3 &
CUDA_VISIBLE_DEVICES=1 python3 util_panbie2.py --source $in_dir --out_dir $out_dir --data_part 3/3 

wait

f_num=$(ls -l $out_dir |grep "^-"|wc -l)
echo "all job is done !"
echo "file num: $f_num"
endTime_s=`date +%s`
sumTime=$[ $endTime_s - $startTime_s ]
echo "Spend Total:$sumTime seconds"

## 发udp结束报文
# echo -en '\xcd\x32\xcd\x32\xcd\x32\x02\x00\x00\x00' | netcat -u 127.0.0.1 11111
python3 end_post.py
