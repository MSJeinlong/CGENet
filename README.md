# CGENet

This repository is the official PyTorch implementation of paper "Context-aware Graph Embedding with Gate and Attention
for Session-based Recommendation"

+ Installation

```commandline
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pyg -c pyg
```

+ Running

```commandline
python main.py --dataset diginetica --k 14 --gnn_layers 3 --gmlp_layers 1
```