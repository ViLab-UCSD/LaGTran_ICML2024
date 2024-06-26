# Tell, Don`t Show!: Language Guidance Eases Transfer Across Domains in Images and Videos (ICML 2024)

Official implementation of the paper [Tell, Don`t Show!: Language Guidance Eases Transfer Across Domains in Images and Videos](https://arxiv.org/abs/2403.05535) published in ICML 2024.

## <center> [[project page](https://tarun005.github.io/lagtran/)] [[paper](https://arxiv.org/abs/2403.05535)] </center>

<div style="text-align:center;">
<img src="assets/BannerPic.gif" alt="LagTrAn" width="75%"/>
</div>


### Abstract
The current standard of unsupervised domain adaptation lacks mechanism for incorporating text guidance. We propose a novel framework called LaGTrAn, which leverages natural language to guide the transfer of discriminative knowledge from labeled source to weakly labeled target domains in image and video classification tasks. Despite its simplicity, LaGTrAn is highly effective on a variety of benchmarks including GeoNet and DomainNet. We also introduce a new benchmark called Ego2Exo to facilitate robustness studies across viewpoint variations in videos, and show LaGTrAn's effeciency in this novel transfer setting. This repository contains the original source code used to train the language and image classifier models in LaGTrAn as well as the trained models. 

<div style="text-align:center;">
<img src="assets/intro.png" alt="teaser_pic" width="55%"/>
</div>

### Requirements

You can use the [requirements.txt](requirements.txt) file to create a new environment or install required packages into existing environments. The following are recommended:
1. Pytorch>=2.0
2. torchvision>=0.14.1
3. timm=0.9.10
4. transformers>=4.30.1
5. tokenizers=0.19.1

### Datasets and metadata

You can access the textual captions and metadata used in our work in the following links. 
1. GeoNet: [GeoPlaces](https://drive.google.com/file/d/11CTqLv6fRCRA3I5u9poZmpFt420XquPx/view?usp=sharing) | [GeoImnet](https://drive.google.com/file/d/1pZmWy4HNl9tGwZErZqckO16DLTLwOMmU/view?usp=sharing) | [GeoUniDA](https://drive.google.com/file/d/1S6T63QMNUc-ZSO_p4rfOxw38uPVZTgrH/view?usp=sharing)
2. DomainNet: [Metadata](https://drive.google.com/file/d/15-7yWWglIT-zAGaeyzxVJj3kwjLTeHsK/view?usp=sharing). 

Download the metadata and places them inside a folder named `metadata`. You can download the original images from the respective webpages: [GeoNet](https://tarun005.github.io/GeoNet/) and [DomainNet](http://ai.bu.edu/M3SDA/). 

### Ego2Exo: A new video adaptation benchmark.

We leverage the recently proposed [Ego-Exo4D](https://docs.ego-exo4d-data.org/) dataset to create a new benchmark called Ego2Exo to study ego-exo transfer in videos. Ego2Exo contains videos from both egocentric and exocentric viewpoints, and is designed to facilitate robustness studies across viewpoint variations in videos. Please refer to [this page](Ego2Exo/README.md) for dataset and metadata for Ego2Exo benchmark.  

### Training.

The training for LagTrAn proceeds in two phases - where we first train the text classifier module, and then use the pseudo-labels derived from that to train the image classification module. Note that the metadata 

Text Classification on 

1. GeoNet: 
```
python3 text_classification.py --dataset [GeoPlaces|GeoImnet] --source usa --target asia --root_dir <data_dir>
```

2. DomainNet:
```
python3 text_classification.py --dataset DomainNet --source real --target clipart --root_dir <data_dir>
```

The trained BERT checkpoints along with the pseudo-labels should be download into `bert_checkpoints` and `pseudo_labels` respectively. The pseudo labels can then be used to train the downstream adaptation network as follows. 

Domain Adaptation on GeoPlaces:
```
python3 train.py --config configs/lagtran.yml --source usa --target asia --dataset [GeoPlaces|GeoImnet] --data_root <data_dir> --exp_name <exp_name> --trainer lagtran
```

Domain Adaptation on DomainNet:
```
python3 train.py --config configs/lagtran.yml --source real --target clipart --dataset DomainNet --data_root <data_dir> --exp_name <exp_name> --trainer lagtran
```

### Trained Models.

You can directly download the target-adapted models (along with the training logs) for GeoNet dataset at the following links. All models use a ViT-B/16 backbone pre-trained on ImageNet and trained using LagTrAn. 

|   |  USA -> Asia | Asia -> USA |
|---|---|---|
| GeoPlaces | 56.14 [(Link)](https://drive.google.com/drive/folders/1QtWuexlXqMskRmPkDt3dUEZmzwR8Oct4?usp=sharing) | 57.02 [(Link)](https://drive.google.com/drive/folders/1XREOIm4bqMDVJZPspbur635eI7SpTF9K?usp=sharing) |
| GeoImnet | 63.67 [(Link)](https://drive.google.com/drive/folders/1ZnnCyHnZnSEuXb8Ygv-IULvUA0QHFpzQ?usp=sharing) | 64.16 [(Link)](https://drive.google.com/drive/folders/1gOPso_6cosfxPFzpbY1QcKuvFhiWXWN3?usp=sharing) |

#### Testing.

If you just want to compute the accuracy using the pre-trained models, you may download the models and use the following command. 

```
python3 test.py --config configs/test.yml --target asia --data_root <data_dir> --saved_model <checkpoint_dir>/best_model.pkl  --dataset [GeoPlaces|GeoImnet]
```

### Citation

If this code or our work helps in your work, please consider citing us. 
``` text
@article{kalluri2024lagtran,
        author    = {Kalluri, Tarun and Majumder, Bodhisattwa and Chandraker, Manmohan},
        title     = {Tell, Don`t Show! Language Guidance Eases Transfer Across Domains in Images and Videos},
        journal   = {ICML},
        year      = {2024},
        url       = {https://arxiv.org/abs/2403.05535},
      },
```

### Contact

If you have any question about this project, please contact [Tarun Kalluri](sskallur@ucsd.edu).