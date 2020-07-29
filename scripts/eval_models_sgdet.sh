#!/usr/bin/env bash

# This is a script that will evaluate all the models for SGDET
export PYTHONPATH=/home/***/code/scene_graph_gen/OS-SGG:/home/***/lib/python_lib/coco/PythonAPI
SAVE_MODEL_PATH="/mnt/data1/***/datasets/visual_genome/model/OS-SGG/"
LOAD_MODEL_PATH="/mnt/data1/***/datasets/visual_genome/model/OS-SGG/"
if [ $1 == "0" ]; then
    echo "EVALING THE BASELINE"
    python models/eval_rels.py -m sgdet -model motifnet -nl_obj 0 -nl_edge 0 -b 6 \
    -clip 5 -p 100 -pooling_dim 4096 -ngpu 1 -ckpt checkpoints/baseline-sgdet/vgrel-17.tar \
    -nepoch 50 -use_bias -cache baseline_sgdet.pkl -test
elif [ $1 == "1" ]; then
    echo "EVALING MESSAGE PASSING"

    python models/eval_rels.py -m sgdet -model stanford -b 6 -p 100 -lr 1e-3 -ngpu 1 -clip 5 \
    -ckpt checkpoints/stanford-sgdet/vgrel-18.tar -cache stanford_sgdet.pkl -test
elif [ $1 == "2" ]; then
    echo "EVALING MOTIFNET"
    python models/eval_rels.py -m sgdet -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet-sgdet/vgrel-14.tar -nepoch 50 -cache motifnet_sgdet.pkl -use_bias
elif [ $1 == "3" ]; then
    echo "EVALING GKG SGDET"
    export CUDA_VISIBLE_DEVICES=0
    echo "CUDA_VISIBLE_DEVICES: "${CUDA_VISIBLE_DEVICES}
    MODEL_NAME_SGDET="*****"
    MODEL_IND="****.tar"
    python3 -u models/eval_rel.py -m sgdet -model gkg -order leftright \
        -nl_obj 0  -nh_obj 0 -nl_obj_gcn 0 -nl_edge 12 -nh_edge 12 -nl_edge_gcn 4 \
        -pooling_dim 4096 -hidden_dim 768 -use_bias \
        -b 1 -clip 5\
        -lr 5e-3 -ngpu 1 -test \
        -ckpt ./checkpoints/${MODEL_NAME}/${MODEL_IND} \
        -nepoch 50 -p 100 \
        -bg_num_rel -1 -test_alpha 1.0;
fi



