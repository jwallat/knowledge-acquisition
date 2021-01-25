#!/bin/bash
# mkdir test_dir
python run_probing.py \
        --decoder_initialization=random \
        --model_type=t5-small \
        --max_epochs=1 \
        --lr 5e-3 \
        --do_probing \
        --batch_size=2 \
        --probing_layer 12 \
        --probing_batch_size=1 \
        --probing_data_dir data/probing_data/ \
        --gpus 1 \
        --output_base_dir=data/outputs/probe_bert/ \
        --run_name bert_layer_12 \
        --train_file data/training_data/wikitext-2-raw/wiki.train.raw \
        --valid_file data/training_data/wikitext-2-raw/wiki.valid.raw \
        --test_file data/training_data/wikitext-2-raw/wiki.test.raw \
        --do_training \
        --use_full_wiki \
        --full_wiki_cache_dir=/home/jonas/git/knowledge-probing/data/training_data/wikitext-2-raw \
        # --use_original_model
        # --fast_dev_run \
        # --use_wandb_logging \
        # --wandb_project_name=probe_bert \
        # --wandb_run_name bert_layer_12 \