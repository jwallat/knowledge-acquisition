#!/bin/bash
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0,1,2,3
gpus=4
pt_seed=123
let ft_seed=$pt_seed+321
pt_bs=8                                                                     # The batch size used during pretraining
pt_ac=4                                                                     # The batches of gradient accumulated during pretraining
ft_bs=64                                                                    # ............. fine-tuning
ft_ac=2                                                                     # ............. fine-tuning
mask_strategy="ssm"                                                         # "norm", "ssm", "pmi"
train_file="./data/PAQ/paq_train_10k.tsv&&./data/PAQ/paq10kunc_train.jsonl" # pretraining
valid_file="/data/PAQ/paq_valid_10k.tsv"
test_file="./data/PAQ/paq_test_10k.tsv"
FT_train_file="./data/PAQ/paq10kunc_train.jsonl"                            # fine-tuning
FT_valid_file="./data/PAQ/paq10kunc_valid.jsonl"
FT_test_file="./data/PAQ/paq10kunc_dev.jsonl"                               # probing
run_name="tloss_SSM_multitask_10kunc"
ft_run_name="tloss_SSM_multitask_10kunc_FT"
wandb_date="2010_"
echo "gpus: $gpus, device: $CUDA_VISIBLE_DEVICES"
echo "pt_seed: $pt_seed, ft_seed: $ft_seed"
echo "pt_bs, pt_ac: $pt_bs, $pt_ac"
echo "ft_bs, ft_ac: $ft_bs, $ft_ac"
echo "train_file: $train_file"
echo "valid_file: $valid_file"
echo "test_file: $test_file"
echo "FT_train_file: $FT_train_file"
echo "FT_valid_file: $FT_valid_file"
echo "FT_test_file: $FT_test_file"
echo "Pipeline name: $run_name, $ft_run_name"
echo "Pipeline is starting!"
source ./PT_multi.sh
value=$(<$run_name".txt")
ckpt_dir=$value
echo "PT_ckpt_dir: $ckpt_dir"
echo "Fine-tuning starting"
source ./FT_multi.sh







