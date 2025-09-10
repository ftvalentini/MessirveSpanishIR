#!/bin/bash -e

CORPUS_DIR=${1:-"data/miracl-corpus/miracl-corpus-v1.0-es"}
OUT_DIR=${2:-"runs/indexes/lucene-index.miracl-v1.0-es"}
N_THREADS=${3:-8}

mkdir -p $OUT_DIR

echo "[bash] Indexing $CORPUS_DIR to $OUT_DIR"

python -m pyserini.index.lucene --collection MrTyDiCollection \
  --input $CORPUS_DIR \
  --index $OUT_DIR \
  --generator DefaultLuceneDocumentGenerator \
  --threads $N_THREADS --storePositions --storeDocvectors \
  --storeRaw -language "es"

