#!/bin/bash -e 
# 
# ...

MODEL_NAME=${1:-"text-embedding-3-large"}
SUBSETS="mx us co es ar pe ve cl gt ec cu bo do hn sv py ni cr pa pr uy no_country full"

# Run if the output directory does not exist
for subset in $SUBSETS; do
    outdir=runs/embeddings/queries_${subset}_test.${MODEL_NAME}

    if [ ! -d $outdir ]; then
        echo "[bash] Encoding $subset test queries with $MODEL_NAME"
        python scripts/get_openai_embeddings.py \
            --dataset spanish-ir/messirve --subset $subset --split test \
            --model_name $MODEL_NAME \
            --outdir $outdir \
            --fields query --id_field id \
            --batch_size 2048 --drop_duplicates

    else
        echo "[bash] $outdir already exists. Skipping."
    fi

done
