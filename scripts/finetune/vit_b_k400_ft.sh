#!/usr/bin/env bash
set -x

# Set a random MASTER_PORT to avoid conflicts
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

# Set default values for variables
OUTPUT_DIR="${OUTPUT_DIR:-/kaggle/working/}"
DATA_PATH="${DATA_PATH:-/kaggle/input/small-csv/small_csv}"
MODEL_PATH="${MODEL_PATH:-/kaggle/input/pretrainedvideomae/vit_g_hybrid_pt_1200e_ssv2_ft.pth}"
JOB_NAME="${JOB_NAME:-vit_h_hybrid_pt_1200e_ssv2_ft}"
PARTITION="${PARTITION:-video}"
GPUS="${GPUS:-32}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-10}"
SRUN_ARGS="${SRUN_ARGS:-}"
PY_ARGS="${@:2}"

# Adjust batch size based on available GPU memory
BATCH_SIZE=4

# Run the training script directly
python -u run_class_finetuning.py \
     --model vit_huge_patch16_224 \
     --data_set SSV2 \
     --nb_classes 6 \
     --data_path "$DATA_PATH" \
     --finetune "$MODEL_PATH" \
     --log_dir "$OUTPUT_DIR" \
     --output_dir "$OUTPUT_DIR" \
     --batch_size "$BATCH_SIZE" \
     --num_sample 2 \
     --input_size 224 \
     --short_side_size 224 \
     --save_ckpt_freq 10 \
     --num_frames 16 \
     --opt adamw \
     --lr 5e-4 \
     --num_workers 10 \
     --opt_betas 0.9 0.999 \
     --weight_decay 0.05 \
     --drop_path 0.2 \
     --layer_decay 0.8 \
     --epochs 20 \
     --test_num_segment 2 \
     --test_num_crop 3 \
     --dist_eval \
     --enable_deepspeed \
     $PY_ARGS