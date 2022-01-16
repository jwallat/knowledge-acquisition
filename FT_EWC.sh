#!/bin/bash
lr=1e-3
warmup_step=0
total_steps=0
echo "lr= $lr"
echo "warmup_step= $warmup_step" 
echo "total_steps= $total_steps"
MASTER_ADDR=localhost WORLD_SIZE=4 NODE_RANK=0 LOCAL_RANK=0 python ft_probing.py\
          --decoder_initialization pre-trained --do_training --do_probing --probing_layer=24 --model_type=t5-base\
          --lr=$lr\
          --seed=$ft_seed\
          --max_steps=100000\
          --gpus=$gpus\
          --batch_size=8\
          --accumulate_grad_batches=4\
          --ft_batch_size=$ft_bs\
          --ft_accumulate_grad_batches=$ft_ac\
          --ft_max_epochs=100\
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
          --training_early_stop_patience=15\
          --use_adafactor \
          --adafactor_relative_step \
          --adafactor_warmup \
          --adafactor_scale_params \
          --use_wandb_logging \
          --wandb_project_name='t5' \
          --wandb_run_name=$wandb_date$ft_run_name\
          --run_name=$ft_run_name\
          --python_executable='/usr/bin/python3'\
          --probing_batch_size=32\
          --accelerator='dp'\
          --num_sanity_val_steps=0\
          --load_model_ckpt_path=$ckpt_dir\
          --mask_way=$mask_strategy\
          --ewc=$use_ewc\
          --ewc_lambda=$ewcLambda\
          #--num_workers=16\



# >> #/home/tzhang/tmp/nlp/knowledge-probing-private/T_11.log #2>&1

