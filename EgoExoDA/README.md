# Ego2Exo: A Video Domain Adaptation Dataset with Natural Language Captions.

## [[Dataset Download](https://drive.google.com/file/d/1pe4F8zYSfA-VYvx296opzHpggrzV-jYE/view?usp=sharing)]

<div style="display:flex;">
    <img src="../assets/ego_Boil_noodles.gif" width="300" style="margin-right: 10px;" />
    <img src="../assets/exo_Boil_noodles.gif" width="300" style="margin-left: 10px;" />
</div>

## Dataset Preparation.

Our Ego2Exo benchmark is curated from the [EgoExo4D](https://ego-exo4d-data.org/) dataset released by Meta. A complete documentation of the dataset preparation including selecting video segments, narrations and label set is available at this colab notebook. To start with this, you first have to download the annotations and keystep labels f

## Dataset Download Instructions

The original video segments should be downloaded from the original [EgoExo4D](https://ego-exo4d-data.org/)  website from Meta by following their guidelines. Specifically, you first have to sign a license form to access and download the dataset, which might take upto 48 hours for getting approved. You can download the [cli installer](https://docs.ego-exo4d-data.org/download/) to enable faster downloads. Make sure that the installation works by running `egoexo --help` which should return the documentation guide. 

In our paper, we only used the pre-extracted [Omnivore features](https://docs.ego-exo4d-data.org/data/features/) for all the video segments as input to the classification network. If you only want to download the pre-extracted features for the takes corresponding to our dataset, use the following command. Make sure to provide the argument `data_dir` or specify where the files should be downloaded.
```
sh omnivore_features_download.sh <data_dir>
```

You can also download the complete videos corresponding to our takes. The whole dataset is quite large, so it is recommended that you only download the videos used in our benchmark with the following command. Make sure to provide the argument `data_dir` or specify where the files should be downloaded. 

```
sh takes_download.sh <data_dir>
```

Finally, the metadata containing the segments, labels and language descriptions for each action segment can be downloaded from this [link](https://drive.google.com/file/d/1pe4F8zYSfA-VYvx296opzHpggrzV-jYE/view?usp=sharing). 

## LagTraAn on Ego2Exo

To run the text classification module on Ego2Exo, use the following command.
```
python3 bert_classification.py --dataset Ego2Exo --source ego --target exo --root_dir <data_dir>
```

Next, we train the video classification module using these pseudo-labels using the following command. Note that we use a smaller learning rate and lesser iterations as we only train the classifier and not the encoder network. 
```
python3 train.py --config configs/lagtran_video_omnivorePT.yml --source exo --target ego --dataset EgoExoDA --data_root <data_dir> --num_iter 45001 --exp_name Ego2Exo_src_exo_tgt_ego --trainer feat_lagtran --batch 64 --val_freq 500
```

### Citation

If this code or the Ego2Exo dataset helps in your work, please consider citing our original paper. 
``` text
@article{kalluri2024lagtran,
        author    = {Kalluri, Tarun and Majumder, Bodhisattwa and Chandraker, Manmohan},
        title     = {Tell, Don`t Show! Language Guidance Eases Transfer Across Domains in Images and Videos},
        journal   = {ICML},
        year      = {2024},
        url       = {https://arxiv.org/abs/2403.05535},
      },
```
