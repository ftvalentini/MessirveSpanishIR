#!/bin/bash -e 

ENCODER_NAME=${1:-"multilingual-e5-large"}
SUBSETS="full no_country mx us co es ar pe ve cl gt ec cu bo do hn sv py ni cr pa pr uy"


# Model specific args:
# To use cosine, normalize_l2 should be always used
model_args=""
if [[ $ENCODER_NAME == "multilingual-e5-large" ]]; then
    model_args="--normalize_l2"
elif [[ $ENCODER_NAME == "mdpr-tied-pft-msmarco-ft-miracl-es" ]]; then
    model_args="" # No L2 normalization
elif [[ $ENCODER_NAME == "text-embedding-3-large" ]]; then
    model_args="--normalize_l2" # because we use cosine similarity
elif [[ $ENCODER_NAME == "multilingual-e5-large-ft-full" ]]; then
    model_args="--normalize_l2"
else
    echo "[bash] Invalid encoder: $ENCODER_NAME"
    exit 1
fi


# Run if the output file does not exist
for subset in $SUBSETS; do
    
    out_file=runs/retrieved/run.eswiki_20240401.${ENCODER_NAME}.messirve-v1.0-$subset-test.tsv

    if [ ! -f $out_file ]; then

        docs_dir=runs/embeddings/eswiki_20240401.${ENCODER_NAME}
        queries_dir=runs/embeddings/queries_${subset}_test.${ENCODER_NAME}

        echo "[bash] Retrieval for: $subset test queries with $ENCODER_NAME"
        python scripts/retrieve.py \
            --doc_embeddings $docs_dir --query_embeddings $queries_dir \
            --out_file $out_file --hits 200 $model_args
    
    else

        echo "[bash] File $out_file already exists. Skipping."

    fi

done
