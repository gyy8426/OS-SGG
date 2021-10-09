# One-shot Scene Graph Generation


This repository contains data and code for the paper “One-shot Scene Graph Generation”. This code is based on the [neural-motifs](https://github.com/rowanz/neural-motifs).

## Framework
<div align=center><img width="800" height="565" src=docs/framework_ver3_00.png/></div>

## Setup


0. Install python3.6 and pytorch 3. I recommend the [Anaconda distribution](https://repo.continuum.io/archive/). To install PyTorch if you haven't already, use
 ```conda install pytorch=0.3.0 torchvision=0.2.0 cuda90 -c pytorch```.
 
1. Update the config file with the dataset paths. Specifically:
    - Visual Genome (the VG_100K folder, image_data.json, VG-SGG.h5, and VG-SGG-dicts.json). See data/stanford_filtered/README.md for the steps I used to download these.
    - ConceptNet. Some files are extracted from the ConceptNet and can be downloaded from [BaiduYun](https://pan.baidu.com/s/1aZNqIY33Hl4tw1SFo329eg) (Password: hj2c)
    - Fix your PYTHONPATH: ```export PYTHONPATH=***/OS-SGG``` or Change Environment Variables in scripts.

2. Compile everything. run ```make``` in the main directory: this compiles the Bilinear Interpolation operation for the RoIs.

3. Pretrain VG detection. The old version involved pretraining COCO as well, but we got rid of that for simplicity. Run ./scripts/pretrain_detector.sh, or download the pretrain model from [GoogleDriver](https://drive.google.com/open?id=11zKRr2OF5oclFL47kjFYBOxScotQzArX).

4. Train OS-SGG: Refer to the scripts ./scripts/train_models_sgcls.sh. Or you can download the trained model from [BaiduYun](https://pan.baidu.com/s/1aZNqIY33Hl4tw1SFo329eg) (Password: hj2c). For the normal scene graph generation task, you should change the load_graphs_one_shot function to the load_graphs function in dataloaders/visual_genome.py
5. Evaluate: Refer to the scripts ./scripts/eval_models_sgcls.sh.
6. Scripts of models are in lib/models/

## Help

Feel free to ping me if you encounter trouble getting it to work!
## Bibtex

```
@inproceedings{sgg:oneshot,
  author    = {Yuyu Guo and
               Jingkuan Song and
               Lianli Gao and
               Heng Tao Shen},
  title     = {One-shot Scene Graph Generation},
  booktitle = {ACM MM},
  pages     = {3090--3098},
  year      = {2020}
}
```
