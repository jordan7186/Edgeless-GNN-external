# Edgeless-GNN-external



Here, we provide additional experimental results in "Edgeless-GNN :Unsupervised Inductive Edgeless Network Embedding".

## Additional experimental results 1

Performance of Edgeless-SAGE on Cora with different number of layers.

|              | AP                | AUC               | macro F1          | micro F1          | NMI               |
| ------------ | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| Single layer | 0.8929 +/- 0.0140 | 0.8905 +/- 0.0127 | 0.6783 +/- 0.0335 | 0.7177 +/- 0.0343 | 0.5109 +/- 0.0212 |
| Two layers   | 0.8464 +/- 0.0142 | 0.8590 +/- 0.0137 | 0.6254 +/- 0.0290 | 0.6665 +/- 0.0267 | 0.4408 +/- 0.0540 |
| Three layers | 0.7329 +/- 0.0221 | 0.7443 +/- 0.0310 | 0.4392 +/- 0.0549 | 0.5177 +/- 0.0422 | 0.3354 +/- 0.0415 |

## Additional experimental results 2

Comparison of different architectures on Citeseer dataset.

| Architecture  | AP                | AUC               | macro F1          | micro F1          | NMI               |
| ------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| Edgeless-SAGE | 0.9394 +/- 0.0006 | 0.9318 +/- 0.0072 | 0.5675 +/- 0.0378 | 0.6502 +/- 0.0299 | 0.4489 +/- 0.0506 |
| Edgeless-GCN  | 0.8921 +/- 0.0131 | 0.8892 +/- 0.0112 | 0.2554 +/- 0.0240 | 0.4943 +/- 0.0423 | 0.2695 +/- 0.0590 |
| Edgeless-GIN  | 0.8633 +/- 0.0146 | 0.8752 +/- 0.0124 | 0.5567 +/- 0.0370 | 0.6687 +/- 0.0362 | 0.2775 +/- 0.0395 |



## Additional experimental results 3

Effect of alpha and beta on Citeseer dataset.

![Additional experiment 3](/add_citeseer.PNG)


## Additional experimental results 4

Comparison with [40] on node classification. We have used the author's implementation with modification to 1) Edge deletion mechanism (to generate edgeless nodes) 2) Train/val/test split to match our setting.

| Dataset        | Method        | micro F1          |
|----------------|---------------|-------------------|
| Cora           | Edgeless-SAGE | 0.7177 +/- 0.0343 |
|                | LDS-GNN       | 0.2777 +/- 0.0693 |
| Citeseer       | Edgeless-SAGE | 0.6697 +/- 0.0299 |
|                | LDS-GNN       | 0.4791 +/- 0.1367 |
