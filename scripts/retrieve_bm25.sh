#!/bin/bash -e 


SPLIT=${1:-"test"}
HITS=${2:-200}
SUBSETS="mx us co es ar pe ve cl gt ec cu bo do hn sv py ni cr pa pr uy no_country full"
INDEX_DIR=runs/indexes/lucene-index.eswiki_20240401


# Run if the output file does not exist
for subset in $SUBSETS; do

    topics_file=data/messirve-v1.0/topics.messirve-v1.0-$subset-$SPLIT.tsv
    out_file=runs/retrieved/run.eswiki_20240401.bm25.messirve-v1.0-$subset-$SPLIT.tsv

    if [ ! -f $out_file ]; then
        python -m pyserini.search.lucene  \
            --index $INDEX_DIR --topics $topics_file --output $out_file \
            --bm25 --language es --threads 8 --hits $HITS --batch-size 128
    else
        echo "[bash] $out_file already exists. Skipping."
    fi

done

echo "[bash] Done!"
