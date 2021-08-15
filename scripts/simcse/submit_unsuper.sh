#!/bin/bash
currentPath="$( cd "$( dirname "$0"  )" && pwd  )"
cd ../../
pwdPath="$(pwd)"

task_dir=/code/NLPDataset/senteval_cn
pretrained_model_path=/code/pre_trained_model/model/chinese-bert-wwm

function execute {
python3 -u -m src.example.simcse.unsuper \
    --task_dir $task_dir \
    --task_name ATEC \
    --pretrained_model_path $pretrained_model_path \
    --epoch 100 \
    --batch_size 64 \
    --learning_rate 1e-5 \
    --max_seq_length 64 \
    --pooler_type cls | tee $pwdPath/log/$(date -d "today" +"%Y%m%d-%H%M%S")_simcse_unsuper.log
}

starttime=`date +'%Y-%m-%d %H:%M:%S'`
execute
endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo '==================================================='
echo "the job execute time： "$((end_seconds-start_seconds))"s"
echo '==================================================='