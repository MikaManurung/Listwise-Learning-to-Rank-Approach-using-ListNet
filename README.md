# Pairwise Learning to Rank Approach Using Light Gradient Boosting Machine (LightGBM)

Learning to Rank (LTR) untuk Information Retrieval (IR) adalah sebuah tugas dalam pembangunan model pemeringkatan menggunakan data pelatihan, sehingga model tersebut dapat mengurutkan objek baru sesuai dengan relevansi, preferensi, atau kepentingannya.  Salah satu pendekatan untuk membangun model _Learning to Rank_ yaitu _pairwise approach_ . _Pairwise approach_ mengambil pasangan dokumen sebagai contoh untuk pelatihan, dan menyusunnya ke LTR sebagai klasifikasi. Terdapat keuntungan pendekatan pairwise yaitu metodologi yang ada pada klasifikasi dapat diterapkan secara langsung dan contoh pelatihan pasangan dokumen dapat dengan mudah diperoleh dalam skenario tertentu. 

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
number of threads : 2
verbose : 10
early stopping rounds : 50
...
```

### Hasil Evaluasi 

|Fold | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|-----|-------------|-------------|-------------|-------------|
|  1  | 0.664544    | 0.702462     | 0.742164    | 0.782247    |
|  2  | 0.679487    | 0.738718    | 0.770269    | 0.803714    |
|  3  | 0.675159    | 0.700883	  | 0.736809    | 0.778698    |
|  4  | 0.660297    | 0.659529    | 0.710893    | 0.758665    | 
|  5  | 0.630573	  | 0.681832    | 0.702157    | 0.766917    | 
|  Rata-rata | 0.662012	  | 0.6966848    | 0.7324584    | 0.7780482    | 

Rata - rata skor NDCG tertinggi yaitu pada NDCG@10 : 0.7780482.

Rata-rata dari  NDCG@1, NDCG@3, NDCG@5 dan NDCG@10  kemudian dijumlahkan dan dicari lagi rata-ratanya sehingga diperoleh skor NDCG keseluruhan dari kelima fold yaitu 
sebesar 0.717301.

Skor NDCG ini membuktikan bahwa penerapan LGBMRanker dan pendekatan _pairwise_ sebagai pendekatan untuk membangun model _Learning to Rank_, menghasilkan skor NDCG yang baik untuk setiap 5 Fold partisi dataset LETOR 4.0 [MQ2008](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/#!letor-4-0).


Referensi 

Chung, K. 2019, Introduction to Learning to Rank, url : https://everdark.github.io/k9/notebooks/ml/learning_to_rank/learning_to_rank.html

