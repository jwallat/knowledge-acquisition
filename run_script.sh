#!/bin/bash
python run_script.py --do_training --decoder_type Huggingface_pretrained_decoder --fast_dev_run=False --gpus 1 --max_epochs=1 --batch_size=2 \
                     --do_probing \
                     --seed 1234 --lr 1e-5
                    #  --use_wandb_logging --wandb_project_name pl0.8_test --python_executable ~/anaconda3/envs/pl_test/bin/python



# python run_script.py --do_probing --probing_model BertForMaskedLM


# python run_script.py --do_probing --probing_model Huggingface_pretrained_decoder