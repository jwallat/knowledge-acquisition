#!/bin/bash
python info_overlap.py \
        --do_training \
        --decoder_initialization random \ 
        --model_type bert-base-uncased \
        --lr=5e-3 \
        --do_probing \
        --batch_size=2 \
        --probing_layer=12 \
        --probing_batch_size=1 \
        --probing_data_dir=data/probing_data/ \
        --gpus=1 \
        --output_base_dir=data/outputs/probe_bert/ \
        --run_name=info_overlap \
        --train_file=data/training_data/wikitext-2-raw/wiki.train.raw \
        --valid_file=data/training_data/wikitext-2-raw/wiki.valid.raw \
        --test_file=data/training_data/wikitext-2-raw/wiki.test.raw 