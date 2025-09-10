#!/bin/bash -e 
# 
# ...


ENCODER=${1:-"intfloat/multilingual-e5-large"}
SUBSETS="full no_country mx us co es ar pe ve cl gt ec cu bo do hn sv py ni cr pa pr uy"

# Model-specific arguments
model_args=""
if [[ $ENCODER == "intfloat/multilingual-e5-large" ]]; then
    model_args="--tf32 --fp16 --pooling mean --prefix query:"
elif [[ $ENCODER == "castorini/mdpr-tied-pft-msmarco-ft-miracl-es" ]]; then
    model_args="--pooling cls --tf32 --fp16"
elif [[ $ENCODER == runs/messirve_training/models/multilingual-e5-large-ft* ]]; then
    model_args="--tf32 --fp16 --pooling mean --prefix query:"
else
    echo "[bash] Invalid encoder: $ENCODER"
    exit 1
fi

encoder_name="${ENCODER##*/}"

# Run if the output directory does not exist
for subset in $SUBSETS; do
    outdir=runs/embeddings/queries_${subset}_test.${encoder_name}

    if [ ! -d $outdir ]; then
        echo "[bash] Encoding $subset test queries with $ENCODER"
        python scripts/encode.py \
            --dataset spanish-ir/messirve --subset $subset --split test \
            --encoder $ENCODER \
            --outdir $outdir \
            --fields query --id_field id \
            --batch_size 4096 --max_gpus 1 \
            --drop_duplicates $model_args
    else
        echo "[bash] $outdir already exists. Skipping."
    fi

done
