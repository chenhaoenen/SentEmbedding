#!/bin/bash
currentPath="$( cd "$( dirname "$0"  )" && pwd  )"
cd ../
pwdPath="$(pwd)"

input_path=$pwdPath/data/book_corpus_small/input.txt
pretrained_model_path=/code/pre_trained_model/model/bart-base

LOG_FILE=$pwdPath/logs/$0.log
rm -f "$LOG_FILE"

run() {
  python -u -m src.example.skip_thought_bart_train \
    --input_path $input_path \
    --epoch 100 \
    --batch_size 128 \
    --max_seq_length 32 \
    --pretrained_model_path $pretrained_model_path  | tee $LOG_FILE
}

s=`date +'%Y-%m-%d %H:%M:%S'`
run
e=`date +'%Y-%m-%d %H:%M:%S'`
echo '==================================================='
echo "the job start time：$s"
echo "the job  end  time：$e"
echo '==================================================='
