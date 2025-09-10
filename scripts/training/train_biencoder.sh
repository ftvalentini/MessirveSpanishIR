#!/bin/bash -e
# 
# NOTE we use --gradient_checkpointing (https://github.com/microsoft/unilm/issues/1120#issuecomment-1584032051)
# NOTE --add_pooler True means a linear layer in the end to change dimensionality of the output
#   (https://github.com/microsoft/unilm/issues/1423#issuecomment-1878365390)
#   (https://github.com/microsoft/unilm/blob/78b3a48de27c388a0212cfee49fd6dc470c9ecb5/simlm/src/models/biencoder_model.py#L40)

SUBSET=${1:-"full"}
ENCODER="intfloat/multilingual-e5-large"
DATA_DIR="runs/messirve_training" # with passages.jsonl.gz

train_data="runs/messirve_training/data/train_neg40_${SUBSET}.jsonl"
dev_data="runs/messirve_training/data/dev_neg40_${SUBSET}.jsonl"

SCRIPTS_DIR="$( cd "$( dirname "$0" )" && pwd )"

encoder_name="${ENCODER##*/}"
output_dir="runs/messirve_training/checkpoints/${encoder_name}_${SUBSET}_$(date +%F-%H%M.%S)"

echo "[bash] Script directory: ${SCRIPTS_DIR}"
echo "[bash] Output directory: ${output_dir}"

mkdir -p "${output_dir}"

deepspeed --num_nodes 1 --num_gpus 2 ${SCRIPTS_DIR}/src/train_biencoder.py --deepspeed ${SCRIPTS_DIR}/ds_config.json \
    --model_name_or_path $ENCODER \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 32 \
    --train_n_passages 8 \
    --add_pooler False \
    --pool_type avg \
    --query_prefix "query:" \
    --doc_prefix "passage:" \
    --t 0.01 \
    --seed 33 \
    --do_train \
    --fp16 \
    --train_file $train_data \
    --validation_file $dev_data \
    --q_max_len 64 \
    --p_max_len 256 \
    --dataloader_num_workers 1 \
    --num_train_epochs 2 \
    --learning_rate 1e-5 \
    --use_scaled_loss True \
    --warmup_steps 400 \
    --share_encoder True \
    --logging_steps 50 \
    --output_dir "${output_dir}" \
    --data_dir "${DATA_DIR}" \
    --save_total_limit 2 \
    --save_strategy epoch \
    --evaluation_strategy steps \
    --eval_steps 0.05 \
    --remove_unused_columns False \
    --overwrite_output_dir \
    --disable_tqdm True \
    --gradient_checkpointing True \
    --report_to none "$@"


echo "[bash] Done!"
