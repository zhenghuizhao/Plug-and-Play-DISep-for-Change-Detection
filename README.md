# Plug-and-Play DISep: Separating Dense Instances for Weakly-Supervised Change Detection
## :notebook_with_decorative_cover: Code for Paper: Plug-and-Play DISep: Separating Dense Instances for Scene-to-Pixel Weakly-Supervised Change Detection in High-Resolution Remote Sensing Images [[arXiv]]()

<p align="center">
    <img src="./tutorials/introduction1.pdf" width="95%" height="95%">
</p>

## Abastract <p align="justify">
<blockquote align="justify"> DISep is designed to separate overlapping dense instances for weakly-supervised change detection. It ensures more accurate counting of change instances, when only knowing that a scene has changed. DIsep can serve as a plug-and-play module, integrating with any baseline pipelines, without adding any inference overhead.


## :speech_balloon: DISep Overview:
<p align="center">
    <img src="./tutorials/method1.pdf" width="95%" height="95%">
</p>


##
## A. Preparations
### 1. Dataset Structure 
``` bash
WSCD dataset with image-level labels:
├─A
├─B
├─label
├─imagelevel_labels.npy
└─list
```

### 2.Create and activate conda environment

```bash
conda create --name transwcd python=3.6
conda activate transwcd
pip install -r requirments.txt
```

##
## B. Train and Test
```bash
# train 
python train.py

```
You can adjust the corresponding implementation settings (e.g., in `WHU.yaml`) for different datasets.


###
```bash
# test
python test.py
```
Please remember to modify the corresponding configurations in `test.py`.

##
## C. Performance
<p align="center">
    <img src="./tutorials/experiment1.png" width="95%" height="95%">
</p>

## Citation
If it's helpful to your research, please kindly cite. Here is an example BibTeX entry:

``` bibtex

```



