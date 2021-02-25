#!/usr/bin/env bash

# This is a script that will train all of the models for scene graph classification and then evaluate them.
export PYTHONPATH=/home/***/code/scene_graph_gen/OS-SGG:/home/***/lib/python_lib/coco/PythonAPI
SAVE_MODEL_PATH="/mnt/data1/***/datasets/visual_genome/model/OS-SGG/"
LOAD_MODEL_PATH="/mnt/data1/***/datasets/visual_genome/model/OS-SGG/"
if [ $1 == "0" ]; then
    echo "TRAINING THE BASELINE"
    python models/train_rels.py -m sgcls -model motifnet -nl_obj 0 -nl_edge 0 -b 6 \
    -clip 5 -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/vgdet/vg-24.tar -save_dir checkpoints/baseline2 \
    -nepoch 50 -use_bias
elif [ $1 == "1" ]; then
    echo "TRAINING MESSAGE PASSING"

    python models/train_rels.py -m sgcls -model stanford -b 6 -p 100 -lr 1e-3 -ngpu 1 -clip 5 \
    -ckpt checkpoints/vgdet/vg-24.tar -save_dir checkpoints/stanford2
elif [ $1 == "2" ]; then
    echo "TRAINING MOTIFNET"
    python models/train_rels.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/vgdet/vg-24.tar \
        -save_dir checkpoints/motifnet2 -nepoch 50 -use_bias
elif [ $1 == "3" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "TRAINING One-Shot CUDA_VISIBLE_DEVICES: "${CUDA_VISIBLE_DEVICES}
    MODEL_NAME="nledge12_nhedge12_edgegcn4_conceptgcn2Glove_drop3ex2mat_sum_drop6cptmat_dirso_sum_dropgcn_BiasDrop3_lr5e3_b16_vgg_oneshot"
    mkdir ./checkpoints/${MODEL_NAME}/
    python3 -u models/train_rels.py -m sgcls -model gkg -order leftright \
        -nl_obj 0  -nh_obj 0 -nl_obj_gcn 0 -nl_edge 12 -nh_edge 12 -nl_edge_gcn 4 \
        -b 16 -clip 5 \
        -pooling_dim 4096 -hidden_dim 768 \
        -lr 5e-3 -ngpu 1 \
        -ckpt ./checkpoints_best/vgdet/vg-faster-rcnn.tar \
        -save_dir  ./checkpoints/${MODEL_NAME}/ \
        -nepoch 6 -p 100 -use_bias \
        -bg_num_rel 2 # -debug -val_size 20 -p 20 -b 3
elif [ $1 == "4" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "TRAINING Normal CUDA_VISIBLE_DEVICES: "${CUDA_VISIBLE_DEVICES}
    MODEL_NAME="nlobj6_nhobj6_nledge6_nhedge6_BiasDrop3_lr5e3_b16_vgg_dot_normal_wmspk_len3"
    mkdir ./checkpoints/${MODEL_NAME}/
    python3 -u models/train_rels_1.py -m sgcls -model gkg -order leftright \
        -nl_obj 6  -nh_obj 6 -nl_obj_gcn 0 -nl_edge 6 -nh_edge 6 -nl_edge_gcn 4 \
        -pooling_dim 4096 -hidden_dim 768 -use_bias \
        -b 16 -clip 5 \
        -lr 5e-3 -ngpu 1 \
        -ckpt ./checkpoints_best/vgdet/vg-faster-rcnn.tar \
        -save_dir  ./checkpoints/${MODEL_NAME}/ \
        -nepoch 6 -p 100 \
        -bg_num_rel 2 # -debug -val_size 20 -p 20 -b 16
fi



