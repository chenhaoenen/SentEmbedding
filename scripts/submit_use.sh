#!/bin/bash
currentPath="$( cd "$( dirname "$0"  )" && pwd  )"
cd ../
pwdPath="$(pwd)"

model_name=en_use_lg
tf_hub_url="https://tfhub.dev/google/universal-sentence-encoder-large/5"
model_url="https://storage.googleapis.com/tfhub-modules/google/universal-sentence-encoder-large/5.tar.gz"

sha1=`python -m src.utils.sha1 $tf_hub_url`

export TFHUB_CACHE_DIR=$pwdPath/data/use/tfhub_models/$model_name

cache_path=$TFHUB_CACHE_DIR/$sha1
mkdir -p $cache_path
wget -c -P $cache_path $model_url
tar -xvf $cache_path/*tar.gz -C $cache_path

python -m src.example.use_eval --model_name $model_name

echo "Finished"
