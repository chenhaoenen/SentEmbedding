#!/bin/bash
currentPath="$( cd "$( dirname "$0"  )" && pwd  )"
cd ../
pwdPath="$(pwd)"

input_path=$pwdPath/data/book_corpus_small/input.txt
pretrained_word2vec_path=/code/NLPDataset/word2vec/GoogleNews-vectors-negative300.bin

LOG_FILE=$pwdPath/logs/$0.log
rm -f "$LOG_FILE"

run() {
  python -u -m src.example.quick_thought_train \
    --input_path $input_path \
    --epoch 100 \
    --batch_size 256 \
    --neg_num 5 \
    --pretrained_word2vec_path $pretrained_word2vec_path  | tee $LOG_FILE
}

s=`date +'%Y-%m-%d %H:%M:%S'`
run
e=`date +'%Y-%m-%d %H:%M:%S'`
echo '==================================================='
echo "the job start time：$s"
echo "the job  end  time：$e"
echo '==================================================='