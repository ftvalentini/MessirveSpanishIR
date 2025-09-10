

### Make a dataset of all queries to analyze topics with BERTopic
python scripts/make_all_queries_csv.py --out_file runs/all_datasets_queries.csv

### Encode all queries (dense) to later run BERTopic
nohup python scripts/encode.py \
    --dataset runs/all_datasets_queries.csv \
    --encoder "intfloat/multilingual-e5-large" \
    --outdir runs/embeddings/queries_all_datasets.multilingual-e5-large \
    --fields query --id_field id \
    --batch_size 2048 --max_gpus 1 \
    --tf32 --fp16 --pooling mean --prefix query: \
    > logs/encode_queries_all_datasets_e5.log 2>&1 &

### BERTopic
METHOD="kmeans" && N=30 &&
nohup python scripts/run_bertopic.py  \
    --dataset runs/all_datasets_queries.csv --embeddings runs/embeddings/queries_all_datasets.multilingual-e5-large  \
    --outdir runs/bertopic_${METHOD}_${N} \
    --cluster_method $METHOD --n $N \
    > logs/bertopic_${METHOD}_${N}.log 2>&1 &
