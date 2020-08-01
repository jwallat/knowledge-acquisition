Inspired by https://github.com/facebookresearch/LAMA


Setup that is required to run the script: 
- Create conda env and install requirements
- Fill in the paths in the config file (in knowledge_probing/config/)
- For a flexible and highly configurable experiment, run the run_script.py with the desired arguments
- To probe single layers, use probe_layer.py with the desired arguments


For creating the plots:
- conda install -c plotly plotly-orca==1.2.1 psutil requests
- the script expects layer data from probes layers in data/layer_data/{default, qa, qa_trained}
- run produce_plots.py
- plot will be in data/layer_data/plots



Todo: 
- Move probing into the pytorch-lightning module and feed datasets as additional dataloaders to trainer.test(trex_dataloader) etc.
- Clean up results json structure a bit (to accessing some values an additional [0] is necessary: data['Google_RE']['place_of_birth'][0]['p_at_1'])