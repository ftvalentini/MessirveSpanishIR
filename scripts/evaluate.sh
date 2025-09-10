#!/bin/bash -e 
# 
# NOTE if data is large, is too slow and uses lots of memory

MODEL_NAMES="bm25 multilingual-e5-large mdpr-tied-pft-msmarco-ft-miracl-es text-embedding-3-large multilingual-e5-large-ft-full"
SUBSETS="full no_country mx us co es ar pe ve cl gt ec cu bo do hn sv py ni cr pa pr uy"
OUTDIR=runs/evaluations

mkdir -p $OUTDIR

# Run evaluation if the results file does not exist:
for model_name in $MODEL_NAMES; do

    for subset in $SUBSETS; do

        run_file=runs/retrieved/run.eswiki_20240401.${model_name}.messirve-v1.0-$subset-test.tsv
        qrels_file=data/messirve-v1.0/qrels.messirve-v1.0-$subset-test.tsv
        out_file=$OUTDIR/eval.eswiki_20240401.${model_name}.messirve-v1.0-$subset-test.tsv

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
