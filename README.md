# MoGE
This is the official implementation of our BIBM 2024 paper "MoGE: Mixture of Graph Experts for Cross-subject Emotion Recognition via Decomposing EEG".

## Abstract
Decoding emotions of previously unseen subjects from electroencephalography (EEG) signals is challenging due to the inter-subject variability. Domain Generalization (DG) methods aim to mitigate the domain shift among different subjects. Once trained, a DG model can be directly deployed on new subjects without any calibration phase. While existing DG studies on cross-subject emotion recognition mainly focus on the design of loss function for domain alignment or regularization, we introduce Sparse Mixture of Graph Experts (MoGE) model to explore DG issues from a new perspective, i.e. the design of the neural architecture. In the MoGE model, routers allocate each EEG channel to a specialized expert, thereby facilitating the decomposition of the intricate brain into distinct functional areas. Extensive experiments on three public datasets demonstrate that compared to other DG methods, our MoGE model trained with empirical risk minimization (ERM) achieves the state-of-the-art (SOTA) accuracies, 88.0%, 74.3%, and 81.8% on SEED, SEED- IV, and SEED-V datasets, respectively.

## Requirements
* python==3.10.9
* pytorch==2.0.0
* timm==0.4.12

* ## Example
Example code for the use of MoGE:
```python
import torch
from torch import nn
from MoGE import Sparse_MoGE

model = Sparse_MoGE(in_channels=62, hidden_channels=62, num_points=5, time_window=5, num_layers=4, heads=2, dim_head=4, num_classes=3, num_experts=3, pool='cls')
# EEG shape = (N, T, C, V), N is batch size, T is window size, C is channel number, V is frequency band number
eeg = torch.rand(size=(32, 5, 62, 5))
output = model(eeg)
```

## Citation
If you find our paper/code useful, please consider citing our work:
```
@inproceedings{liu2023moge,
  title={MoGE: Mixture of Graph Experts for Cross-subject Emotion Recognition via Decomposing {EEG}},
  author={Liu, Xuan-Hao and Jiang, Wei-Bang and Zheng, Wei-Long and Lu, Bao-Liang},
  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year={2024},
  organization={IEEE}
}
```
