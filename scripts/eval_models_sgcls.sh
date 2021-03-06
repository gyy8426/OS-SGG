#!/usr/bin/env bash

# This is a script that will evaluate all models for SGCLS
export PYTHONPATH=/home/***/code/scene_graph_gen/OS-SGG:/home/***/lib/python_lib/coco/PythonAPI
SAVE_MODEL_PATH="/mnt/data1/***/datasets/visual_genome/model/OS-SGG/"
LOAD_MODEL_PATH="/mnt/data1/***/datasets/visual_genome/model/OS-SGG/"
if [ $1 == "0" ]; then
    echo "EVALING THE BASELINE"
    python models/eval_rels.py -m sgcls -model motifnet -nl_obj 0 -nl_edge 0 -b 6 \
    -clip 5 -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/baseline-sgcls/vgrel-11.tar \
    -nepoch 50 -use_bias -test -cache baseline_sgcls
    python models/eval_rels.py -m predcls -model motifnet -nl_obj 0 -nl_edge 0 -b 6 \
    -clip 5 -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/baseline-sgcls/vgrel-11.tar \
    -nepoch 50 -use_bias -test -cache baseline_predcls
elif [ $1 == "1" ]; then
    echo "EVALING MESSAGE PASSING"
    python models/eval_rels.py -m sgcls -model stanford -b 6 -p 100 -lr 1e-3 -ngpu 1 -clip 5 \
    -ckpt checkpoints/stanford-sgcls/vgrel-11.tar -test -cache stanford_sgcls
    python models/eval_rels.py -m predcls -model stanford -b 6 -p 100 -lr 1e-3 -ngpu 1 -clip 5 \
    -ckpt checkpoints/stanford-sgcls/vgrel-11.tar -test -cache stanford_predcls
elif [ $1 == "2" ]; then
    echo "EVALING MOTIFNET"
    python models/eval_rels.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet-sgcls/vgrel-7.tar -nepoch 50 -use_bias -cache motifnet_sgcls
    python models/eval_rels.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet-sgcls/vgrel-7.tar -nepoch 50 -use_bias -cache motifnet_predcls
elif [ $1 == "3" ]; then
    echo "EVALING One-Shot"
    export CUDA_VISIBLE_DEVICES=0
    echo "CUDA_VISIBLE_DEVICES: "${CUDA_VISIBLE_DEVICES}
    MODEL_NAME="nledge12_nhedge12_edgegcn4_conceptgcn2Glove_drop3ex2mat_sum_drop6cptmat_dirso_sum_dropgcn_BiasDrop3_lr5e3_b16_vgg_oneshot"
    MODEL_IND="vgrel-5.tar"
    python3 -u models/eval_rels.py -m predcls -model gkg -order leftright \
        -nl_obj 0  -nh_obj 0 -nl_obj_gcn 0 -nl_edge 12 -nh_edge 12 -nl_edge_gcn 4 \
        -pooling_dim 4096 -hidden_dim 768 -use_bias \
        -b 1 -clip 5\
        -lr 5e-3 -ngpu 1 -test \
        -ckpt ./checkpoints/${MODEL_NAME}/${MODEL_IND} \
        -nepoch 50 -p 100 \
        -bg_num_rel -1 -test_alpha 1.0;
    python3 -u models/eval_rels.py -m sgcls -model gkg -order leftright \
        -nl_obj 0  -nh_obj 0 -nl_obj_gcn 0 -nl_edge 12 -nh_edge 12 -nl_edge_gcn 4 \
        -pooling_dim 4096 -hidden_dim 768 -use_bias \
        -b 1 -clip 5\
        -lr 5e-3 -ngpu 1 -test \
        -ckpt ./checkpoints/${MODEL_NAME}/${MODEL_IND} \
        -nepoch 50 -p 100 \
        -bg_num_rel -1 -test_alpha 1.0;
    python3 -u models/eval_rels.py -m sgdet -model gkg -order leftright \
        -nl_obj 0  -nh_obj 0 -nl_obj_gcn 0 -nl_edge 12 -nh_edge 12 -nl_edge_gcn 4 \
        -pooling_dim 4096 -hidden_dim 768 -use_bias \
        -b 1 -clip 5\
        -lr 5e-3 -ngpu 1 -test \
        -ckpt ./checkpoints/${MODEL_NAME}/${MODEL_IND} \
        -nepoch 50 -p 100 \
        -bg_num_rel -1 -test_alpha 1.0;
elif [ $1 == "4" ]; then
    echo "EVALING Normal"
    export CUDA_VISIBLE_DEVICES=0
    echo "CUDA_VISIBLE_DEVICES: "${CUDA_VISIBLE_DEVICES}
    MODEL_NAME="nlobj6_nhobj6_nledge6_nhedge6_BiasDrop3_lr5e3_b16_vgg_dot_normal_wmspk_len3"
    MODEL_IND="vgrel-16.tar"
    MODEL_NAME_SGDET="nlobj6_nhobj6_nledge6_nhedge6_BiasDrop3_lr5e3_b16_vgg_dot_normal_wmspk_len3_lr5e4_b6_sgdet"
    MODEL_IND_SGDET="vgrel-30.tar"
    python3 -u models/eval_rels_1.py -m predcls -model gkg -order leftright \
        -nl_obj 6  -nh_obj 6 -nl_obj_gcn 0 -nl_edge 6 -nh_edge 6 -nl_edge_gcn 4 \
        -pooling_dim 4096 -hidden_dim 768 -use_bias \
        -b 1 -clip 5\
        -lr 5e-3 -ngpu 1 -test \
        -ckpt ./checkpoints/${MODEL_NAME}/${MODEL_IND} \
        -nepoch 50 -p 100 \
        -bg_num_rel -1 -test_alpha 1.0;
    python3 -u models/eval_rels_1.py -m sgcls -model gkg -order leftright \
        -nl_obj 6  -nh_obj 6 -nl_obj_gcn 0 -nl_edge 6 -nh_edge 6 -nl_edge_gcn 4 \
        -pooling_dim 4096 -hidden_dim 768 -use_bias \
        -b 1 -clip 5\
        -lr 5e-3 -ngpu 1 -test \
        -ckpt ./checkpoints/${MODEL_NAME}/${MODEL_IND} \
        -nepoch 50 -p 100 \
        -bg_num_rel -1 -test_alpha 1.0;
    python3 -u models/eval_rels_1.py -m sgdet -model gkg -order leftright \
        -nl_obj 6  -nh_obj 6 -nl_obj_gcn 0 -nl_edge 6 -nh_edge 6 -nl_edge_gcn 4 \
        -b 1 -clip 5\
        -pooling_dim 4096 -hidden_dim 768 \
        -lr 5e-3 -ngpu 1 -test \
        -ckpt ./checkpoints/${MODEL_NAME_SGDET}/${MODEL_IND_SGDET} \
        -nepoch 50 -p 100 -use_bias \
        -bg_num_rel -1 -test_alpha 1.0;
fi



