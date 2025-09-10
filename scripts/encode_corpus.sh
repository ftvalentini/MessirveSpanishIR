#!/bin/bash -e
# 
# ... 

ENCODER=${1:-"intfloat/multilingual-e5-large"}
CORPUS_DIR=data/eswiki_20240401_corpus

# Model-specific arguments
model_args=""
if [[ $ENCODER == "intfloat/multilingual-e5-large" ]]; then
    model_args="--pooling mean --prefix passage: --tf32 --fp16"
elif [[ $ENCODER == "castorini/mdpr-tied-pft-msmarco-ft-miracl-es" ]]; then
    model_args="--pooling cls --tf32 --fp16"
elif [[ $ENCODER == runs/messirve_training/models/multilingual-e5-large-ft* ]]; then
    model_args="--pooling mean --prefix passage: --tf32 --fp16"
else
    echo "[bash] Invalid encoder: $ENCODER"
    exit 1
fi

# Run:
echo "[bash] Encoding corpus with $ENCODER"
encoder_name="${ENCODER##*/}"
outdir=runs/embeddings/eswiki_20240401.${encoder_name}
python scripts/encode.py \
    --dataset $CORPUS_DIR --encoder $ENCODER --outdir $outdir \
    --batch_size 512 --max_gpus 1 \
    $model_args

echo "[bash] Done!"
