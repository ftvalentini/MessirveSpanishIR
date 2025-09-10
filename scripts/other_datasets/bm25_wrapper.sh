#!/bin/bash -e

DATASETS=(
    "pres"
    "mmarco"
    "miracl"
    "sqac"
    "meup"
)

for dataset in "${DATASETS[@]}"; do
    
    # Indexing
    CORPUS_DIR=data/${dataset}_corpus
    INDEX_DIR=runs/indexes/lucene-index.${dataset}

    if [ ! -d "$INDEX_DIR" ]; then
        echo "[bash] Indexing: $dataset"
        scripts/create_lucene_index.sh $CORPUS_DIR $INDEX_DIR
    else
        echo "Directory '$INDEX_DIR' already exists. Skipping..."
    fi


    # Retrieval
    topics_file=data/${dataset}_eval/topics.${dataset}.tsv
    run_file=runs/retrieved/run.${dataset}.bm25.${dataset}.tsv

    if [ ! -f "$run_file" ]; then
        echo "[bash] Retrieving: $dataset with ${topics_file}"
        python -m pyserini.search.lucene  \
            --index $INDEX_DIR --topics $topics_file --output $run_file \
            --bm25 --language es --threads 8 --hits 200 --batch-size 128
    else
        echo "Run file '$run_file' already exists. Skipping..."
    fi


done

echo "[bash] DONE!"
