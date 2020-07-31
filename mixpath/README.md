# MixPath: A Unified Approach for One-shot Neural Architecture Search

This repo provides the supernet of S<sub>1</sub> and our confirmatory experiments on NAS-Bench-101.

```
Python >= 3.6, Pytorch >= 1.0.0, torchvision >= 0.2.0
```
## mixpath 

2020.7.31 代码阅读

## Datasets

CIFAR-10 can be automatically downloaded by `torchvision`. It has 50,000 images for
training and 10,000 images for validation.

## Usage

```
python S1/train_search.py \
    --exp_name experiment_name \
    --m number_of_paths[1,2,3,4]
    --data_dir /path/to/dataset \
    --seed 2020 \
```
```
python NasBench101/nas_train_search.py \
    --exp_name experiment_name \
    --m number_of_paths[1,2,3,4]
    --data_dir /path/to/dataset \
    --seed 2020 \
```


