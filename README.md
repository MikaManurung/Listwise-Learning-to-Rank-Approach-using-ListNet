# Pairwise Learning to Rank Approach Using Light Gradient Boosting Machine (LightGBM)
_Learning to Rank_ adalah bagian dari penerapan machine learning dalam pembangunan model perankingan pada sistem temu balik informasi. Salah satu pendekatan untuk membangun model _Learning to Rank_ yaitu _pairwise approach_ dimana pasangan dokumen dianggap sebagai input untuk sistem pembelajaran dan untuk meningkatkan kinerja model yang akan dibangun diterapkan algoritma _gradient tree boosting_. Salah satu implementasi _gradient tree boosting_ terbaik yang tersedia saat ini adalah XGBoost (_eXtreme Gradient Boosting_).

Learning to Rank (LTR) untuk Information Retrieval (IR) adalah sebuah tugas dalam pembangunan model pemeringkatan menggunakan data pelatihan, sehingga model tersebut dapat mengurutkan objek baru sesuai dengan relevansi, preferensi, atau kepentingannya [1]Tie-Yan Liu (2009).  Salah satu pendekatan untuk membangun model _Learning to Rank_ yaitu _pairwise approach_ . _Pairwise approach_ mengambil pasangan dokumen sebagai contoh untuk pelatihan, dan menyusunnya ke LTR sebagai klasifikasi. Terdapat keuntungan pendekatan pairwise yaitu metodologi yang ada pada klasifikasi dapat diterapkan secara langsung dan contoh pelatihan pasangan dokumen dapat dengan mudah diperoleh dalam skenario tertentu[2](Joachims, 2002). 

Kemudian kami menggunakan LightGBM yang  merupakan singkatan dari _Light Gradient Boosting Machine_ adalah _framework_ peningkatan gradien yang didasarkan pada algoritma decision tree dan digunakan untuk peringkat, klasifikasi, dan tugas pembelajaran mesin lainnya. Fokus pengembangannya adalah pada kinerja dan skalabilitas. LightGBM memiliki banyak keunggulan seperti pengoptimalan sparse, parallel training, multiple loss functions, regularisasi, bagging, dan penghentian awal.

Evaluasi performa perankingan Learning to Rank menggunakan daftar peringkat Normalized Discounted Cumulative Gain (NDCG).


### Dataset
Million queries dataset from TREC 2008 :
[MQ2008](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/#!letor-4-0)

| Dataset name |       | rows   | columns | num samples in queries (min, median, max) | 
|--------------|-------|--------|---------|-------------------------------------------| 
| mq2008       | train | 9630   | 46      | (5, 8, 121)                               | 
|              | test  | 2874   | 46      | (6, 14, 119)                              | 

### Parameter
Untuk membangun model perankingan digunakan XGBRanker dimana parameter yang digunakan adalah sebagai berikut:

```
...
num_leaves : 255
min_data_in_leaf : 1
min_sum_hessian_in_leaf : 100
objective": regression
eval_metric : 'ndcg'
learning_rate: 0.1
...
```


Rata - rata skor NDCG kelima Fold : 

Skor NDCG  kelima fold atau rata-rata skor NDCG setiap _fold_ yaitu  dari 1. Skor NDCG ini membuktikan bahwa penerapan LGBMRanker dan pendekatan _pairwise_ sebagai pendekatan untuk membangun model _Learning to Rank_, menghasilkan skor NDCG yang baik untuk setiap 5 Fold partisi dataset LETOR 4.0 [MQ2008](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/#!letor-4-0).
