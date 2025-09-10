#!/bin/bash -e
# 
# NOTE running sequentially multiple times because in each new run
# previous results are reordered according to frequency
# NOTE see scripts/google_questions/helpers.py for the list of countries


COUNTRIES=("mx" "us" "co" "es" "ar" "pe" "ve" "cl" "gt" "ec" "cu" "bo" "do" "hn" "sv" "py" "ni" "cr" "pa" "pr" "uy")
# MAX_RESULTS=${1:-50000}


for country in "${COUNTRIES[@]}"; do
    
    # this will consider 1 word after the prefix
    for max_results in 170000 190000; do
        date=$(date '+%d-%m-%Y_%H-%M-%S')
        python -u scripts/google_questions/extract_google_questions.py \
            --outdir runs/google_questions \
            --country_code "${country}" \
            --min_words_to_add 1 --max_words_to_add 2 \
            --max_results $max_results \
            >> logs/questions_${country}_${date}.log 2>&1
    done

    # this will consider 2 words after the prefix
    for max_results in 210000 230000; do
        date=$(date '+%d-%m-%Y_%H-%M-%S')
        python -u scripts/google_questions/extract_google_questions.py \
            --outdir runs/google_questions \
            --country_code "${country}" \
            --min_words_to_add 2 --max_words_to_add 3 \
            --max_results $max_results \
            >> logs/questions_${country}_${date}.log 2>&1
    done

done

