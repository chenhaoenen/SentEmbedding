#!/bin/bash
currentPath="$( cd "$( dirname "$0"  )" && pwd  )"
cd ../
pwdPath="$(pwd)"


vocab_dir=/code/SentEmbedding/data/gensen/vocab

LOG_FILE=$pwdPath/logs/$0.log
rm -f "$LOG_FILE"

run() {
  python -u -m src.example.gensen_train \
    --vocab_dir $vocab_dir | tee $LOG_FILE
}

s=`date +'%Y-%m-%d %H:%M:%S'`
run
e=`date +'%Y-%m-%d %H:%M:%S'`
echo '==================================================='
echo "the job start time：$s"
echo "the job  end  time：$e"
echo '==================================================='