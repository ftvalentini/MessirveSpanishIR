#!/bin/bash -e 
# 
# NOTE if data is large, is too slow and uses lots of memory

DATASETS="pres mmarco miracl sqac meup"
MODEL_NAMES="bm25 multilingual-e5-large mdpr-tied-pft-msmarco-ft-miracl-es multilingual-e5-large-ft-full"
OUTDIR=runs/evaluations

mkdir -p $OUTDIR

# Run evaluation if the results file does not exist:
for dataset in $DATASETS; do
    for model_name in $MODEL_NAMES; do

        run_file=runs/retrieved/run.${dataset}.${model_name}.${dataset}.tsv
        qrels_file=data/${dataset}_eval/qrels.$dataset.tsv
        out_file=$OUTDIR/eval.${dataset}.${model_name}.${dataset}.tsv

        if [ ! -f $run_file ]; then
            echo "[bash] $run_file does not exist. Skipping."
            continue
        
        elif [ ! -f $out_file ]; then
            echo "[bash] Evaluating $run_file"
            python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 -m recall.100 $qrels_file $run_file \
                > >(tee $out_file) 2>&1

        else
            echo "[bash] $out_file already exists. Skipping."
        fi


    done
done
