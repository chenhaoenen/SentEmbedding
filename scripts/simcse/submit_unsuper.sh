#!/bin/bash
currentPath="$( cd "$( dirname "$0"  )" && pwd  )"
cd ../../
pwdPath="$(pwd)"

task_dir=/code/NLPDataset/senteval_cn
pretrained_model_path=/code/pre_trained_model/model/chinese-bert-wwm

execute() {
  echo "task name: $1; Shell Temperature: $2";
  python3 -u -m src.example.simcse.unsuper \
      --task_dir $task_dir \
      --task_name $1 \
      --pretrained_model_path $pretrained_model_path \
      --epoch 2 \
      --batch_size 64 \
      --learning_rate 1e-5 \
      --max_seq_length 64 \
      --pooler_type cls \
      --log_freq 10000 \
      --temp $2 | tee $pwdPath/log/$(date -d "today" +"%Y%m%d-%H%M%S")_simcse_unsuper.log
}



starttime=`date +'%Y-%m-%d %H:%M:%S'`
tasks=(ATEC BQ LCQMC PAWSX STS-B)

temps=(0.0001 0.0005 0.001 0.005 0.008 0.01 0.03 0.05 1)

for task in ${tasks[@]};
do
  for temp in ${temps[@]};
  do
      execute $task $temp;
      echo "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
  done
done

endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo '==================================================='
echo "the job execute time： "$((end_seconds-start_seconds))"s"
echo '==================================================='