#!/bin/bash
export TOKENIZERS_PARALLELISM=true

export CUDA_VISIBLE_DEVICES=0,1,2,3
run_name="baseline_10K" #_only_FT&PonPAQ_web
lr=1e-3
warmup_step=0
total_steps=0
echo "lr= $lr"
echo "warmup_step= $warmup_step" 
echo "total_steps= $total_steps"
echo "Cool name is $run_name"

MASTER_ADDR=localhost WORLD_SIZE=4 NODE_RANK=0 LOCAL_RANK=0 python ft_probing.py\
          --decoder_initialization pre-trained --do_probing --probing_layer=24 --model_type=t5-base\
          --lr=$lr\
          --max_steps=100000\
          --gpus=1\
          --batch_size=12\
          --accumulate_grad_batches=16\
          --unfreeze_transformer\
          --warmup_steps=$warmup_step\
          --total_steps=$total_steps\
          --train_file="./"\
          --valid_file="./"\
          --test_file="./"\
          --FT_train_file="./data/PAQ/paq10k_train.jsonl"\
          --FT_valid_file="./data/PAQ/paq10k_valid.jsonl"\
          --FT_test_file="./data/PAQ/paq10k_dev.jsonl"\
          --training_early_stop_delta=0.0001\
          --training_early_stop_patience=15\
          --use_adafactor \
          --adafactor_relative_step \
          --adafactor_warmup \
          --adafactor_scale_params \
          --use_wandb_logging \
          --wandb_project_name='t5' \
          --wandb_run_name='1211_baseline_10K'\
          --run_name=$run_name\
          --python_executable='/usr/bin/python3'\
          --probing_batch_size=32\



# >> #/home/tzhang/tmp/nlp/knowledge-probing-private/T_11.log #2>&1

