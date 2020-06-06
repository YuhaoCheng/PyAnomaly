## Model Zoo
We provide the model for different dataset and different methods

### Video Dataset

#### Ped2 Baseline

| Method             | AUC  | Download |
| ------------------ | ---- | -------- |
| STAE[1]            | 91.2 |          |
| MemAE[2]           | 94.1 |          |
| OCAE[5]            | 97.8 |          |
| AnoPCN[7]          | 96.8 |          |
| LTR[8]             | 90.0 |          |
| MLAD[9]            | 99.2 |          |
| sRNN[4]            | 92.2 |          |
| sRNN-AE[20]        | 92.2 |          |
| AICN[12]           | 95.3 |          |
| AMC[13]            | 96.2 |          |
| AnoPred[14]        | 95.4 |          |
| MPED-RNN[15]       | -    |          |
| LSA[19]            | 95.4 |          |
| OCC[16]            | -    |          |
| ITAE[21]           | -    |          |
| Unmasking[3]       | 82.2 |          |
| discriminative[10] | -    |          |
| TwoStage[11]       | 96.4 |          |

#### Avenue Baseline

| Method             | AUC  | Download |
| ------------------ | ---- | -------- |
| STAE[1]            | 77.1 |          |
| MemAE[2]           | 83.3 |          |
| OCAE[5]            | 90.4 |          |
| AnoPCN[7]          | 86.2 |          |
| LTR[8]             | 70.2 |          |
| MLAD[9]            | 71.5 |          |
| sRNN[4]            | 81.7 |          |
| sRNN-AE[20]        | 83.5 |          |
| AICN[12]           | 77.2 |          |
| AMC[13]            | 86.9 |          |
| AnoPred[14]        | 85.1 |          |
| MPED-RNN[15]       | -    |          |
| LSA[19]            | -    |          |
| OCC[16]            | -    |          |
| ITAE[21]           | -    |          |
| Unmasking[3]       | 80.6 |          |
| discriminative[10] | 78.3 |          |
| TwoStage[11]       | 85.3 |          |




#### ShanghaiTech Baseline

| Method             | AUC  | Download |
| ------------------ | ---- | -------- |
| STAE[1]            | -    |          |
| MemAE[2]           | 71.2 |          |
| OCAE[5]            | 84.9 |          |
| AnoPCN[7]          | 73.6 |          |
| LTR[8]             | -    |          |
| MLAD[9]            | -    |          |
| sRNN[4]            | 68.0 |          |
| sRNN-AE[20]        | 69.3 |          |
| AICN[12]           | -    |          |
| AMC[13]            | -    |          |
| AnoPred[14]        | 72.8 |          |
| MPED-RNN[15]       | 73.4 |          |
| LSA[19]            | 72.5 |          |
| OCC[16]            | -    |          |
| ITAE[21]           | 72.5 |          |
| Unmasking[3]       | -    |          |
| discriminative[10] | -    |          |
| TwoStage[11]       | -    |          |

### Image Dataset

#### MNIST Baseline

| Method        | AUC  | Download |
| ------------- | ---- | -------- |
| MemAE[2]      | 97.5 |          |
| CoRA[6]       | -    |          |
| OCC[16]       | -    |          |
| ITAE[21]      | 98.3 |          |
| OCGAN[17]     | 97.5 |          |
| Deep SVDD[18] | -    |          |

#### CIFAR-10 Baseline
| Method        | AUC  | Download |
| ------------- | ---- | -------- |
| MemAE[2]      | 60.9 |          |
| CoRA[6]       | 89.7 |          |
| OCC[16]       | -    |          |
| ITAE[21]      | 86.6 |          |
| OCGAN[17]     | 65.7 |          |
| Deep SVDD[18] | -    |          |

## Reference

[1] Spatio-Temporal AurtoEncoder for video anomaly detection(MM2017)

[2] Memoryizing Normality to detect anomaly: memory-augmented deep Autoencoder for Unsupervised anomaly detection(iccv2019

[3] Unmasking the abnormal events in videos(ICCV2017)

[4] A revisit of Sparse coding Based Anomaly detection in Stacked RNN framework(ICCV2017) (shanghai tech)

[5] Object-centric autocoders and dummy anomalies for abnormal event detection in video(cvpr2019)

[6] Learning competitive and Discriminative Reconstructions for Anomaly Detection(AAAI2019)

[7] AnoPCN: Video anomaly detection via deep predictive coding(MM2019)

[8] Learning Temporal Regularity in Video Sequences(CVPR2016)

[9] Robust Anomlay Dtection in Videos using Multilevel Representations(AAAI2019)

[10] A discriminative framework for anomaly detection in large videos(ECCV2016)

[11] Detecting Abnormality without Knowing Normality: A Two-stage Approach for Unsupervised Video Abnormal Event Detection(MM2018

[12] Video Anomlay Dtection and Localization Based on an Adaptive Intra-Frame Classification Network(TMM2020)

[13] Anomaly Detection in Video sequence with appearance-motion correspondence(ICCV2019)

[14] Future Frame Predication for Anomaly Detection-A New Baseline(CVPR2018)

[15] Learning Regularity in Skeleton Trajectories for Anomaly Detection in Videos(CVPR2019)

[16] Adversarially Learned One-Class Classifier for Novelty Detection(CVPR2018)

[17] OCGAN: One-class Novelty Detection Using GANs with Constrained Latent Representations(CVPR2019)

[18] Deep One-Class Classification(ICML2018)

[19] Latent Space Autoregression for Novelty Detection(CVPR2019)

[20] Video anomaly detection with sparse coding inspired deep neural networks(TPMI2019)

[21] Inverse-Transform AutoEncoder for Anomaly Detection