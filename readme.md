Setup that is required to run the script: 
- Create conda env and install requirements
- Fill in the paths in the config file
- run the run_script.py with the desired arguments


For creating the plots:
- conda install -c plotly plotly-orca==1.2.1 psutil requests
- the script expects layer data from probes layers in data/layer_data/{default, qa, qa_trained}
- run produce_plots.py
- plot will be in data/layer_data/plots