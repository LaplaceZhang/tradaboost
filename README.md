Transfer Learning
=====
See the original code for TrAdaBoost at [chenchiwei's work](https://github.com/chenchiwei/tradaboost)

Here we use this transfer learning algorithm for intrusion detection purpose, the dataset can find at [CIC IDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html).

Compare with another transfer learning algorithm TCA(transfer component analysis), originaly is from [Jindong Wang's work](https://github.com/jindongwang/transferlearning), we use this for IDS purpose as well. The first I attempt to use dataset contains attack A to train and use it to classify Attack B but failed(one potential reason is the target domain is class imbalanced). So I added some Attack B instances into the source to assist training process, The results of ROC curve:

![](https://github.com/LaplaceZhang/tradaboost/blob/master/ROC%20Curve%20for%20TCA.png)

