#!/bin/bash
python run_knowledge_capacity.py --decoder_initialization=pre-trained \
        --model_type=bert-base-uncased \
        --max_epochs=1 \
        --probing_layer=12 \
        --probing_batch_size=4 \
        --probing_data_dir data/probing_data/ \
        --num_workers=1 \
        --output_base_dir=data/outputs/knowledge_capacity/ \
        --run_name=knowledge_capacity \
        --train_file=data/training_data/wikitext-2-raw/wiki.train.raw \
        --valid_file=data/training_data/wikitext-2-raw/wiki.valid.raw \
        --test_file=data/training_data/wikitext-2-raw/wiki.test.raw \
        --capacity_text_mode=templates \
        --capacity_masking_mode=object \
        --unfreeze_transformer \
        --gpus=1 