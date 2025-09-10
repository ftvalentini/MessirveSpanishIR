

### Download datasets
python -u scripts/download_hf_corpus.py \
    --dataset spanish-ir/eswiki_20240401_corpus --outdir data/eswiki_20240401_corpus
python -u scripts/download_qrels.py \
    --dataset spanish-ir/messirve-trec --outdir data/messirve-v1.0 --pattern "messirve-v1.0"

### BM25 index of corpus
CORPUS_DIR=data/eswiki_20240401_corpus &&
OUT_DIR=runs/indexes/lucene-index.eswiki_20240401 &&
nohup scripts/create_lucene_index.sh $CORPUS_DIR $OUT_DIR > logs/wiki_lucene_index.log 2>&1 &

### Encode corpus (dense)
nohup scripts/encode_corpus.sh intfloat/multilingual-e5-large > logs/encode_corpus_e5_$(date '+%d-%m-%Y_%H-%M-%S').log 2>&1 &
nohup scripts/encode_corpus.sh castorini/mdpr-tied-pft-msmarco-ft-miracl-es > logs/encode_corpus_mdpr-es_$(date '+%d-%m-%Y_%H-%M-%S').log 2>&1 &

### Encode corpus (OpenAI) -- Only for our dataset.
nohup python scripts/get_openai_embeddings.py \
    --dataset data/eswiki_20240401_corpus \
    --model_name "text-embedding-3-large" \
    --outdir runs/embeddings/eswiki_20240401.text-embedding-3-large \
    --batch_size 2048 \
    > logs/encode_text-embedding-3-large_$(date '+%d-%m-%Y_%H-%M-%S').log 2>&1 &

### Encode queries (dense)
nohup scripts/encode_queries.sh intfloat/multilingual-e5-large > logs/encode_queries_e5_$(date '+%d-%m-%Y_%H-%M-%S').log 2>&1 &
nohup scripts/encode_queries.sh castorini/mdpr-tied-pft-msmarco-ft-miracl-es > logs/encode_queries_mdpr-es_$(date '+%d-%m-%Y_%H-%M-%S').log 2>&1 &
nohup scripts/encode_queries_openai.sh > logs/encode_queries_openai_$(date '+%d-%m-%Y_%H-%M-%S').log 2>&1 &

### BM25 retrieval
nohup scripts/retrieve_bm25.sh > logs/retrieval_bm25.eswiki_20240401.log 2>&1 &

### Dense retrieval
nohup scripts/retrieve_dense.sh multilingual-e5-large > logs/retrieval_multilingual-e5-large_$(date '+%d-%m-%Y_%H-%M-%S').log 2>&1 &
nohup scripts/retrieve_dense.sh mdpr-tied-pft-msmarco-ft-miracl-es > logs/retrieval_mdpr-tied-pft-msmarco-ft-miracl-es_$(date '+%d-%m-%Y_%H-%M-%S').log 2>&1 &
nohup scripts/retrieve_dense.sh text-embedding-3-large > logs/retrieval_text-embedding-3-large_$(date '+%d-%m-%Y_%H-%M-%S').log 2>&1 &

### Evaluation metrics
nohup scripts/evaluate.sh > logs/evaluation_$(date '+%d-%m-%Y_%H-%M-%S').log 2>&1 &

### Getting BM25 hard negatives for fine-tuning
nohup scripts/retrieve_bm25.sh "train" 250 > logs/retrieval_bm25.eswiki_20240401.train.log 2>&1 &

### Preparing fine-tuning data:
# Collection:
if [ ! -f "runs/messirve_training/passages.jsonl.gz" ]; then
    nohup python -u scripts/prepare_training_corpus.py \
        --corpus_dir data/eswiki_20240401_corpus --out_dir runs/messirve_training \
        > logs/prepare_training_corpus.log 2>&1 &
fi
# Qrels:
nohup python -u scripts/prepare_training_samples.py \
    --ids_file runs/messirve_training/docid_map.json \
    --trec_dir data/messirve-v1.0 \
    --runs_dir runs/retrieved \
    --out_dir runs/messirve_training/data \
    --n_negatives 40 \
    > logs/prepare_training_samples.log 2>&1 &

### Training
nohup scripts/training/train_biencoder.sh > logs/train_full_$(date '+%d-%m-%Y_%H-%M-%S').log 2>&1 &

### Create soft link from final model:
mkdir -p runs/messirve_training/models &&
cd runs/messirve_training/models/ &&
ln -sf ../checkpoints/multilingual-e5-large_full_2024-10-02-1751.07 multilingual-e5-large-ft-full &&
cd -
# To upload to Hub, see: notebooks/upload_model_to_hf.ipynb

### Encode docs and queries with finetuned model + retrieve + evaluate:
nohup scripts/encode_corpus.sh runs/messirve_training/models/multilingual-e5-large-ft-full > logs/encode_corpus_e5-ft-full_$(date '+%d-%m-%Y_%H-%M-%S').log 2>&1 &
nohup scripts/encode_queries.sh runs/messirve_training/models/multilingual-e5-large-ft-full > logs/encode_queries_e5-ft-full_$(date '+%d-%m-%Y_%H-%M-%S').log 2>&1 &
nohup scripts/retrieve_dense.sh multilingual-e5-large-ft-full > logs/retrieval_e5-ft-full_$(date '+%d-%m-%Y_%H-%M-%S').log 2>&1 &
nohup scripts/evaluate.sh > logs/evaluation_$(date '+%d-%m-%Y_%H-%M-%S').log 2>&1 &


### Evaluation in other datasets ###############################################
### Build datasets with correct format
# Run scripts/other_datasets/build_datasets.ipynb

### BM25 index of corpus + retrieval
TIME=$(date '+%d-%m-%Y_%H-%M-%S')
LOG_FILE=logs/bm25_other_datasets_$TIME.log
nohup scripts/other_datasets/bm25_wrapper.sh > $LOG_FILE 2>&1 &
echo "tail -f $LOG_FILE"

### Encoding with dense models + retrieval
TIME=$(date '+%d-%m-%Y_%H-%M-%S')
LOG_FILE=logs/dense_other_datasets_$TIME.log
nohup scripts/other_datasets/dense_models_wrapper.sh > $LOG_FILE 2>&1 &
echo "tail -f $LOG_FILE"

### Evaluation metrics
TIME=$(date '+%d-%m-%Y_%H-%M-%S')
LOG_FILE=logs/eval_other_datasets_$TIME.log
nohup scripts/other_datasets/evaluate.sh > $LOG_FILE 2>&1 &
echo "tail -f $LOG_FILE"
###############################################################################
