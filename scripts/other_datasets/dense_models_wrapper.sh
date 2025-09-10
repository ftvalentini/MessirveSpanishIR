#!/bin/bash

DATASETS=(
    "pres"
    "mmarco"
    "miracl"
    "sqac"
    "meup"
)

MODELS=(
    "intfloat/multilingual-e5-large"
    "castorini/mdpr-tied-pft-msmarco-ft-miracl-es"
    "runs/messirve_training/models/multilingual-e5-large-ft-full"
)


for dataset in "${DATASETS[@]}"; do

    for model in "${MODELS[@]}"; do

        ### Encode corpus ######################################
        model_args=""
        if [[ $model == "intfloat/multilingual-e5-large" ]]; then
            model_args="--pooling mean --prefix passage: --tf32 --fp16"
        elif [[ $model == "castorini/mdpr-tied-pft-msmarco-ft-miracl-es" ]]; then
            model_args="--pooling cls --tf32 --fp16"
        elif [[ $model == runs/messirve_training/models/multilingual-e5-large-ft* ]]; then
            model_args="--pooling mean --prefix passage: --tf32 --fp16"
        else
            echo "[bash] Invalid model: $model"
            exit 1
        fi

        model_name="${model##*/}"
        corpus_dir=data/${dataset}_corpus
        outdir=runs/embeddings/${dataset}_corpus.${model_name}
        
        if [ ! -e "$outdir" ]; then
            echo "[bash] Encoding $dataset corpus with $model"
            python scripts/encode.py \
                --dataset $corpus_dir --encoder $model --outdir $outdir \
                --batch_size 8 --max_gpus 2 \
                $model_args
        else
            echo "Dir '$outdir' already exists. Skipping..."
        fi
        
        ### Encode queries ######################################
        model_args=""
        if [[ $model == "intfloat/multilingual-e5-large" ]]; then
            model_args="--tf32 --fp16 --pooling mean --prefix query:"
        elif [[ $model == "castorini/mdpr-tied-pft-msmarco-ft-miracl-es" ]]; then
            model_args="--pooling cls --tf32 --fp16"
        elif [[ $model == runs/messirve_training/models/multilingual-e5-large-ft* ]]; then
            model_args="--tf32 --fp16 --pooling mean --prefix query:"
        else
            echo "[bash] Invalid model: $model"
            exit 1
        fi

        model_name="${model##*/}"
        queries_file=data/${dataset}_eval/queries.${dataset}.csv
        outdir=runs/embeddings/queries_${dataset}.${model_name}

        if [ ! -e "$outdir" ]; then
            echo "[bash] Encoding $dataset queries with $model"
            python scripts/encode.py \
                --dataset $queries_file \
                --encoder $model \
                --outdir $outdir \
                --fields query --id_field id \
                --batch_size 16 --max_gpus 2 \
                --drop_duplicates $model_args
        else
            echo "[bash] $outdir already exists. Skipping."
        fi

        ### Retrieve #############################################
        model_name="${model##*/}"
        model_args=""
        if [[ $model_name == "multilingual-e5-large" ]]; then
            model_args="--normalize_l2"
        elif [[ $model_name == "mdpr-tied-pft-msmarco-ft-miracl-es" ]]; then
            model_args="" # No L2 normalization
        elif [[ $model_name == "text-embedding-3-large" ]]; then
            model_args="--normalize_l2" # because we use cosine similarity
        elif [[ $model_name == multilingual-e5-large-ft* ]]; then
            model_args="--normalize_l2"
        else
            echo "[bash] Invalid encoder: $model_name"
            exit 1
        fi

        docs_dir=runs/embeddings/${dataset}_corpus.${model_name}
        queries_dir=runs/embeddings/queries_${dataset}.${model_name}
        out_file=runs/retrieved/run.${dataset}.${model_name}.${dataset}.tsv
        
        if [ ! -f "$out_file" ]; then
            echo "[bash] Retrieval for $dataset with $model_name"
            python scripts/retrieve.py \
                --doc_embeddings $docs_dir --query_embeddings $queries_dir \
                --out_file $out_file --hits 200 $model_args
        else
            echo "[bash] File $out_file already exists. Skipping."
        fi

    done

done

echo "[bash] DONE!"
