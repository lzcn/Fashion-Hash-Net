# Fashion Hash Net

## Description

This responsitory contains the code of paper [Learning Binary Code for Personalized Fashion Recommendation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Lu_Learning_Binary_Code_for_Personalized_Fashion_Recommendation_CVPR_2019_paper.pdf)

## Required Packages

- pytorch
- torchvision
- PIL
- numpy
- pandas
- tqdm
- lmdb
- yaml
- visdom

I upgraded the version of [PyTorch](https://pytorch.org) to `1.2.0` and the package dependency is solved automatically with [`conda`](https://docs.conda.io/en/latest/).

The last 4 packages can be install via `conda`:

```bash
conda install python-lmdb pyyaml visdom tqdm -c conda-forge
```

## How to Use the Code

The main script [`scripts/run.py`](scripts/run.py) currently supports the following functions:

```python
ACTION_FUNS = {
    # train models
    "train": train,
    # runing the FITB task
    "fitb": fitb,
    # evaluate pairs accuracy
    "evaluate-accuracy": evalute_accuracy,
    # evaluate NDCG and AUC
    "evaluate-rank": evalute_rank,
    # compute the binary codes
    "extract-features": extract_features,
}
```

### Configurations

There are three main modules in `polyvore`:

- `polyvore.data`: module for polyvore-dataset
- `polyvore.model`: module for fashion hash net
- `polyvore.solver`: module for training

For configurations, see `polyvore.param`, and we give some examples in `cfg` folder.
The configuration file was written in [yaml](https://pyyaml.org/) format.

### Train

To train `FHN-T3` with both visual and semantic features, run the following script:

```bash
scripts/run.py train --cfg ./cfg/train/FHN_VSE_T3_630.yaml
```

### Evaluate

To evaluate the accuracy of positive-negative pairs:

```bash
scripts/run.py evaluate-accuracy --cfg ./cfg/evalute/FHN_VSE_T3_630.yaml  
```

To evaluate the rank quality:

```bash
scripts/run.py evaluate-rank --cfg ./cfg/evaluate-rank/FHN_VSE_T3_630.yaml
```

To evaluate the FITB task:

```bash
scripts/run.py fitb --cfg ./cfg/fitb/FHN_VSE_T3_630.yaml 
```

## How to Use the Polyvore-$U$s

Download the data from [OneDrive](https://stduestceducn-my.sharepoint.com/:f:/g/personal/zhilu_std_uestc_edu_cn/Er7BPeXpVc5Egl9sufLB7V0BdYVoXDj8PcHqgYe3ze2i-w)

See [data/README.md](data/README.md) for details

### How to Cite

```latex
@inproceedings{Lu:2019tk,
author = {Lu, Zhi and Hu, Yang and Jiang, Yunchao and Chen, Yan and Zeng, Bing},
title = {{Learning Binary Code for Personalized Fashion Recommendation}},
booktitle = {CVPR},
year = {2019}
}
```

## Contact

Email: zhilu@std.uestc.edu.cn