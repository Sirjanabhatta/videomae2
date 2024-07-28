#!/usr/bin/env bash
set -x
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='/kaggle/working/videomae2/checkpoints'
DATA_PATH='/kaggle/input/overall-data/csv_files'
# DATA_PATH='/kaggle/input/csv-files-for-cricshot/crichotcsv'

MODEL_PATH='/kaggle/input/vitbase/vit_b_k710_dl_from_giant.pth'
# MODEL_PATH='/kaggle/input/videomae-epoch17/checkpoint-17.pth'

JOB_NAME=$1
PARTITION=${PARTITION:-"video"}
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-1}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}
# --resume /kaggle/input/videomae-epoch17/checkpoint-17.pth \

# batch_size can be adjusted according to the graphics card
# srun -p $PARTITION \
#         --job-name=${JOB_NAME} \
#         --gres=gpu:${GPUS_PER_NODE} \
#         --ntasks=${GPUS} \
#         --ntasks-per-node=${GPUS_PER_NODE} \
#         --cpus-per-task=${CPUS_PER_TASK} \
#         --kill-on-bad-exit=1 \
#         --async \
python run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set Kinetics-710 \
        --nb_classes 3 \
        --resume /kaggle/input/best-check-videomae/checkpoint-best.pth \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 3 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 10 \
        --opt adamw \
        --lr 0.001 \
        --drop_path 0.1 \
        --head_drop_rate 0.0 \
        --layer_decay 0.75 \
        --opt_betas 0.9 0.999 \
        --warmup_epochs 1 \
        --epochs 50 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --dist_eval 
      