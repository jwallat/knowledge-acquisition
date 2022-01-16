#!/bin/bash
lr=1e-3
warmup_step=0
total_steps=0
echo "lr= $lr"
echo "warmup_step= $warmup_step" 
echo "total_steps= $total_steps"

MASTER_ADDR=localhost WORLD_SIZE=4 NODE_RANK=0 LOCAL_RANK=0 python pretraining_probing.py\
          --decoder_initialization pre-trained --do_training --do_probing --probing_layer=24 --model_type=t5-base\
          --lr=$lr\
          --seed=$pt_seed\
          --max_steps=200000\
          --gpus=$gpus\
          --batch_size=$pt_bs\
          --accumulate_grad_batches=$pt_ac\
          --unfreeze_transformer\
          --warmup_steps=$warmup_step\
          --total_steps=$total_steps\
          --train_file=$train_file\
          --valid_file=$valid_file\
          --test_file=$test_file\
          --FT_train_file=$FT_train_file\
          --FT_valid_file=$FT_valid_file\
          --FT_test_file=$FT_test_file\
          --training_early_stop_delta=0.0001\
          --training_early_stop_patience=10\
          --use_adafactor \
          --adafactor_relative_step \
          --adafactor_warmup \
          --adafactor_scale_params \
          --use_wandb_logging \
          --wandb_project_name='t5' \
          --wandb_run_name=$wandb_date$run_name\
          --run_name=$run_name\
          --python_executable='/usr/bin/python3'\
          --probing_batch_size=32\
          --accelerator='dp'\
          --multitask=True\
          --mask_way=$mask_strategy\
          --pmi_path='./data/pmi_dict_2000k_M.pkl'\
          #--load_model_ckpt_path='./'\
          #--num_workers=16\
          #--use_raw_model\


# >> #/home/tzhang/tmp/nlp/knowledge-probing-private/T_11.log #2>&1

