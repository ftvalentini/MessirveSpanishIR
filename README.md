
Code to replicate [_MessIRve: A Large-Scale Spanish Information Retrieval Dataset_](https://arxiv.org/abs/2409.05994) (Valentini et al., EMNLP 2025).

Cite as:

```bibtex
@inproceedings{valentini-etal-2025-messirve,
    title = "{M}ess{IR}ve: {A} {L}arge-{S}cale {S}panish {I}nformation {R}etrieval {D}ataset",
    author = "Valentini, Francisco  and
      Cotik, Viviana  and
      Furman, Dami{\'a}n  and
      Bercovich, Ivan  and
      Altszyler, Edgar  and
      P{\'e}rez, Juan Manuel",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
}
```


## Set up

Follow instructions in `setup.sh`:

Install git lfs to git clone large files from huggingface:

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash &&
sudo apt-get install git-lfs &&
git lfs install
```

**NOTE**: if you have issues installing `nmslib`, do the following:

```bash
pip install --upgrade pybind11        # 2.10.1 or higher  (latest as of today: 2.11.1)
pip install --verbose  'nmslib @ git+https://github.com/nmslib/nmslib.git#egg=nmslib&subdirectory=python_bindings'
```

Then, re-run `pip install -r requirements.txt`. See [this](https://github.com/nmslib/nmslib/issues/538) for further details.


## Dataset creation

Create a Spanish IR dataset using questions from Google's autocomplete API and answers from Google SERP's featured snippets related to Wikipedia.

### Question extraction

Extract questions from Google autocomplete:

```bash
# wrapper for all countries:
nohup scripts/google_questions/extract_questions_wrapper.sh &> /dev/null &
```

### Answer extraction pipeline

Initialize postgre database + create queries, htmls, extractions tables + populate queries table with Google questions:

```bash
# To start the postgresql service: sudo systemctl start postgresql
python scripts/google_questions/create_tables.py
# NOTE this script can be run multiple times, it will not duplicate data
```

Fetch search URLs that still have to be scraped from the DB, and save one file for each country, with `id\turl` in each line:

```bash
# Remove old urls and then run
rm runs/google_questions/urls/* &&
python scripts/google_questions/fetch_search_urls.py
```

**NOTE** this will overwrite files in `runs/google_questions/urls`, but the files will be the same if no new questions were added to the db or no questions have already been parsed (html extracted).

Then run the scraping of featured snippets using `scripts/scraper.py`, and save scraped htmls to PostgreSQL:

```bash
mkdir -p runs/google_questions/htmls/to_load
mkdir -p runs/google_questions/htmls/loaded
python scripts/google_questions/populate_htmls_table.py \
    --input_dir "runs/google_questions/htmls/to_load" &&
mv runs/google_questions/htmls/to_load/* runs/google_questions/htmls/loaded
# NOTE Can be run safely multiple times, it will not duplicate data
```

Extract featured snippets from htmls and save to PostgreSQL:

```bash
date=$(date '+%d-%m-%Y_%H-%M-%S') &&
nohup python -u scripts/google_questions/extract_answers.py >> logs/htmlparse_${date}.log 2>&1 &
```

### Wikipedia corpus

We build a fresh Wikipedia corpus to get the latest data that can match with the answers we got from Google.

Download dump:

```bash
DUMP_DATE=20240401 &&
wget -c -b -P data/ "https://dumps.wikimedia.org/eswiki/${DUMP_DATE}/eswiki-${DUMP_DATE}-pages-articles-multistream.xml.bz2"
```

Extract jsons with wikiextractor:

```bash
DUMP_DATE=20240401 &&
OUTDIR="runs/google_questions/wikipedia/eswiki-${DUMP_DATE}-extracted" &&
mkdir -p $OUTDIR &&
date=$(date '+%d-%m-%Y_%H-%M-%S') &&
nohup python -m wikiextractor.WikiExtractor "data/eswiki-${DUMP_DATE}-pages-articles-multistream.xml.bz2" \
    --namespaces "Portal,Anexo" --json --bytes 10G --output $OUTDIR > "logs/wiki_extraction_${date}.log" 2>&1 &
```

Create a MIRACL-like corpus (jsonl.gz files of 500_000 documents each):

```bash
DUMP_DATE=20240401 &&
python -u scripts/google_questions/create_eswiki_corpus.py \
    --input_dir "runs/google_questions/wikipedia/eswiki-${DUMP_DATE}-extracted" \
    --output_dir runs/corpora/eswiki-${DUMP_DATE}-corpus
```

### Create IR datasets

Attach wikipedia passages to "valid" (query,answer) pairs:

```bash
date=$(date '+%d-%m-%Y_%H-%M-%S') &&
nohup python -u scripts/google_questions/match_documents.py \
    --corpus_dir "runs/corpora/eswiki-20240401-corpus" \
    --version 6 --max_workers 14 > "logs/match_documents_${date}.log" 2>&1 &
```

Finally:

1. Create `qrels.csv` with dataset splits + upload corupus and data splits to HF with `notebooks/upload_dataset_to_hf.ipynb`.
2. Use `qrels.csv` as input to create topics and qrels tsv files for train and test, and upload to HF (NOTE the repo name is not provided to preserve anonymity).

```bash
python -u scripts/google_questions/save_topics_and_qrels.py \
    --input_file "runs/google_questions/qrels_1.1.csv.gz" \
    --outdir "runs/google_questions/trec_files/messirve-v1.1" \
    --dataset_name "messirve-v1.1" \
    --hf_dir "spanish-ir/messirve-trec"
```

3. Get questions for annotation with `notebooks/questions_analysis.ipynb`.


## Training and evaluations

Follow instructions in `run_experiments.sh`.


## Topic analysis

Follow instructions in `run_topic_analysis.sh` to run BERTopics on the Spanish IR datasets.

Then run `notebooks/topic_analysis.ipynb` to analyze the topics.


## Figures and analysis

To replicate the figures and analysis in the paper, run the following notebooks:

* `notebooks/datasets_stats.ipynb`
* `notebooks/messirve_spanish_varieties.ipynb`
* `notebooks/messirve_stats.ipynb`
* `notebooks/quality_annotations.ipynb`
* `notebooks/models_evaluations.ipynb`
* `notebooks/other_datasets_eval.ipynb`
* `notebooks/query_analysis.ipynb`
* `notebooks/error_analysis.ipynb`
* `notebooks/miracl_analysis.ipynb`


## Others

### Verify pyserini installation

Verify BM25 retrieval works on prebuilt index on NFC dataset:

```bash
python -m pyserini.search.lucene \
    --index beir-v1.0.0-nfcorpus.flat \
    --topics beir-v1.0.0-nfcorpus-test \
    --output tmp_run.beir-v1.0.0-nfcorpus-flat.trec \
    --output-format trec \
    --batch 36 --threads 12 \
    --remove-query --hits 1000

python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 -m recall.100,1000 beir-v1.0.0-nfcorpus-test tmp_run.beir-v1.0.0-nfcorpus-flat.trec
```

Sources:

* https://github.com/castorini/pyserini/blob/e6700f6a1bca7d2bea81fb40d9c3ae63c1be142a/docs/installation.md?plain=1#L75
* https://github.com/castorini/pyserini/blob/e6700f6a1bca7d2bea81fb40d9c3ae63c1be142a/scripts/beir/run_beir_baselines.py#L57

