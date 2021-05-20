#!/bin/bash
if [ -f edb7cbe2-9d73-11e4-aa78-bcaec51b9163_MQ2008.rar ]
then
    echo "Use downloaded data to run experiment."
else
    echo "Downloading data."
    # wget https://s3-us-west-2.amazonaws.com/xgboost-examples/MQ2008.rar
    wget http://www.bigdatalab.ac.cn/benchmark/upload/download_source/edb7cbe2-9d73-11e4-aa78-bcaec51b9163_MQ2008.rar
    unrar x edb7cbe2-9d73-11e4-aa78-bcaec51b9163_MQ2008.rar
    mv -f MQ2008/MQ2008/Fold1/*.txt .
fi

python /content/drive/MyDrive/ColabNotebooks/stbi/proyek/trans_data.py train.txt mq2008.train mq2008.train.group

python /content/drive/MyDrive/ColabNotebooks/stbi/proyek/trans_data.py test.txt mq2008.test mq2008.test.group

python /content/drive/MyDrive/ColabNotebooks/stbi/proyek/trans_data.py vali.txt mq2008.vali mq2008.vali.group