#!/bin/sh

python audiocap/dataset_store.py \
    --dataset-name cahya/audiosnippets-tiny \
    --train-split 0.5 \
    --max-rows 250 \
    --dataset-column-audio mp3 \
    --dataset-column-metadata json \
    --dataset-column-file-name sample_id \
    --dataset-column-duration duration \
    --dataset-column-duration-scale 1.0
