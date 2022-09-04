in_dir=/data/pref_test/yjsk
out_dir=/data/pref_test/yjsk_result

startTime_s=`date +%s`

cd /data/PatrolAi/patrol_ai/python_codes
CUDA_VISIBLE_DEVICES=0 python3 util_yjsk.py --source $in_dir --out_dir $out_dir --data_part 1/3 &
CUDA_VISIBLE_DEVICES=0 python3 util_yjsk.py --source $in_dir --out_dir $out_dir --data_part 2/4 &
CUDA_VISIBLE_DEVICES=1 python3 util_yjsk.py --source $in_dir --out_dir $out_dir --data_part 3/4 &
CUDA_VISIBLE_DEVICES=1 python3 util_yjsk.py --source $in_dir --out_dir $out_dir --data_part 4/4 

wait

f_num=$(ls -l $out_dir |grep "^-"|wc -l)
echo "all job is done !"
echo "file num: $f_num"
endTime_s=`date +%s`
sumTime=$[ $endTime_s - $startTime_s ]
echo "Spend Total:$sumTime seconds"