# Knowledge Probing for T5
---

Knowledge probing is a framework that allows probing for T5's world knowledge. It is based on the closed-book question answering framework by Roberts et. al. ([https://github.com/google-research/google-research/tree/master/t5_closed_book_qa](https://github.com/google-research/google-research/tree/master/t5_closed_book_qa)). This framework investigates the impact of Random Tokens Masking, Salient Span Masking and PMI-Masking on the benefits of additional pre-training as a means of knowledge enhancement. While increasing the overall knowledge reserve of T5, the EWC algorithm and multi-task training are used to reduce T5's forgetting of learned knowledge. Finally, under the dual effects of knowledge enhancement and reduction of knowledge forgetting, the performance of T5 on the CBQA task is improved by an average of 23% compared with the original pre-trained T5. ([https://huggingface.co/t5-base](https://huggingface.co/t5-base))

# Data

We borrowed passages and QA-pairs from the 65 Million Probably Asked Questions (PAQ) dataset ([https://github.com/facebookresearch/PAQ](https://github.com/facebookresearch/PAQ)). for pre-training and fine-tuning, respectively.

![image](https://user-images.githubusercontent.com/52611192/142775232-31151150-5b6b-4434-a565-fad9084b05c4.png)

---

## Getting Started

1. Clone repository
2. Create virtual environment

    ```bash
    $ conda create -n knowledge-probing-cbqa python=3.7 
    ```

3. Install pytorch 1.7.1 (inclusive CUDA if you want to use a GPU)
4. Install requirements:  

    ```bash
    $ pip install -r requirements.txt
    ```

5. Run the setup script in /scripts/ that downloads the data and creates neccessary folders: 

    ```bash
    $ sh setup.sh 
    ```
Please download the PAQ dataset by yourself. Passages and corresponding QA-pairs can be obtained from preprocessed Wikipedia dump and PAQ QA-pair metadata ([https://github.com/facebookresearch/PAQ](https://github.com/facebookresearch/PAQ)).

6. Download the [PMI vocabulary](https://drive.google.com/file/d/14lnHwUjoAXQS6vNsSMjfpJuzXEkKikSB/view?usp=sharing) of Wikipeadia + Bookcorpus. When using the PMI-masking strategy, please use flags:
```bash 
$ python pretraining_probing.py\
      -- pmi_path = .../pmi_dict_2000k_M.pkl \
 ```
Or you can collect PMI vocabulary from other corpora based on the original [PMI-Masking paper](https://arxiv.org/pdf/2010.01825.pdf).

That's it! Run your experiments!

### Implement Experiments

If you want to implement the pipeline of pre-training + fine-tuning with EWC algorithm, you can do like this:

```bash
$ bash pipeline_EWC.sh
```
You can also set related parameters and masking strategies in the PT_EWC.sh and FT_EWC.sh files. 

If you have run pipeline_EWC.sh with a certain masking strategy, the multitask training with this masking strategy can be implemented directly.
Run the pipeline of multitask training:

```bash
$ bash pipeline_multi.sh
```

### Weights and Biases Integration

Training and probing is integrated with Weights and Biases. To log your experiments with W&B, simply add these flags to the program call: 

```bash
$ python run_probing.py \
			... \
			--use_wandb_logging \
			--wandb_project_name probe_bert_model \
```

### Models

Knowledge probing can be done for various pre-trained or fine-tuned T5-models. You can either supply your own models or load models from the huggingface model hub ([https://huggingface.co/models](https://huggingface.co/models)).

Models from the huggingface model hub can easily downloaded:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("<MODEL_NAME_FROM_MODEL_HUB>")
model.save_pretrained('<PATH>')
```

To probe own models, make sure to set the according flag when running the scirpt:

```bash
--use_model_from_dir
```

### QA-pairs.jsonl

Please store QA-pairs in the jsonl file like this:
```python
{"question": "Who is the president of the United States?", "answer": ["Joseph Robinette Biden Jr.", "Joe Biden"], "passage_id": "2114", "subsets": "L4"}
```
## How to reproduce our experiments

Here you can [download](https://drive.google.com/drive/folders/1rkusTS0aPm54yTcGENbUDr9RASt0_lPe?usp=sharing) some of the subsets of PAQ datasets we have processed.
If additional data sets are required please handle them yourself.

Training the baseline model. This means that there is no additional pre-training, only a fine-tuning process on the QA-set dataset. The QA-pair dataset can be set by modifying flags  "FT_train/valid/dev_file".

```bash
$ bash baseline_ft.sh 
```
Standard additional pretraining with different masking strategies → Standard fine-tuning → Test (Probe knowledge).
The masking strategy can be set by modifying the flag "mask_strategy" in the pipeline_EWC.sh file.

```bash
$ bash pipeline_EWC.sh 0 0    # The first parameter is set to 0 in order to avoid using the EWC algorithm. The second parameter does not matter here.
```

Standard additional pretraining with different masking strategies → Fine-tuning with EWC algorithm → Test (Probe knowledge).

```bash 
$ bash pipeline_EWC.sh 1 1000 # Using EWC algorithm, and the scale foctor of EWC is set as 1000 (adjustable).
```
Multitask training → Standard fine-tuning → Test (Probe knowledge).

```bash
$ bash pipeline_multitask.sh 
```
Note: All settings can be customised in the individual .sh files.

## Reference

Will follow, paper not released yet.
