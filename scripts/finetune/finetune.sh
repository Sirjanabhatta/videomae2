#!/usr/bin/env bash
set -x  # print the commands

CUDA_VISIBLE_DEVICES=0
OUTPUT_DIR='/kaggle/working/'
DATA_PATH='/kaggle/input/small-csv/small_csv'
MODEL_PATH='/kaggle/input/pretrainedvideomae/vit_g_hybrid_pt_1200e_ssv2_ft.pth'




# Please refer to `run_class_finetuning.py` for the meaning of the following hyperreferences
torchrun --standalone --nproc_per_node=1 \
          run_class_finetuning.py \
        --model vit_giant_patch14_224 \
        --data_set Kinetics-710 \
        --nb_classes 6 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 1 \
        --input_size 112 \
        --short_side_size 112 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --sampling_rate 8 \
        --num_sample 2 \
        --num_workers 1 \
        --opt adamw \
        --lr 1e-3 \
        --drop_path 0.3 \
        --clip_grad 5.0 \
        --layer_decay 0.9 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.1 \
        --warmup_epochs 5 \
        --epochs 35 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --dist_eval  \


