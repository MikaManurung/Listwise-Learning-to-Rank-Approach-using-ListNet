#!/usr/bin/python
import lightgbm as lgb
from sklearn.datasets import load_svmlight_file

#  Melakukan petangkingan dengan LGBMRanker
x_train, y_train = load_svmlight_file("mq2008.train")
x_valid, y_valid = load_svmlight_file("mq2008.vali")
x_test, y_test = load_svmlight_file("mq2008.test")

# Mendefenisikan data train
group_train = []
with open("mq2008.train.group", "r") as f:
    data = f.readlines()
    for line in data:
        group_train.append(int(line.split("\n")[0]))

# Mendefenisikan data validasi		
group_valid = []
with open("mq2008.vali.group", "r") as f:
    data = f.readlines()
    for line in data:
        group_valid.append(int(line.split("\n")[0]))

# Mendefenisikan data test
group_test = []
with open("mq2008.test.group", "r") as f:
    data = f.readlines()
    for line in data:
        group_test.append(int(line.split("\n")[0]))

# Mendefenisikan Parameter yang digunakan 
param = {
    #"task": "train",
    "num_leaves": 255, #jumlah maksimal leaves yang terdapat pada 1 tree
    "min_data_in_leaf": 1, ##jumlah maksimal leaves yang terdapat pada 1 tree
    "min_sum_hessian_in_leaf": 100,
    "objective": "regression", #spesifikasi model ML yang digunakan
    "metric": "ndcg", #metric evaluasi yang digunakan 
	  # "ndcg_eval_at": [1, 3, 5, 10],
    "learning_rate": 0.1,
}

gbm = lgb.LGBMRanker(**param) # gbm adalah model yang menyimpan LGBMRanker dan parameter yang sudah didefenisikan
gbm.fit(x_train, y_train, group=group_train, verbose=10,
          eval_set=[(x_valid, y_valid)], eval_group=[group_valid], eval_at = [1,3,5,10],
          early_stopping_rounds=50)
          # early_stopping_rounds adalah jumlah iterasi yang dilakukan untuk menemukan nilai terbaik


pred = gbm.predict(x_test)