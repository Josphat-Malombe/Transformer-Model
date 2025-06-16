from pathlib import Path
def get_config():
    return{
        "batch_size":8,
        "lr":0.0001,
        "d_model":512,
        "seq_len":350,
        "num_epochs":20,
        "lang_src":"en",
        "lang_tgt":"it",
        "model_basename":"tmodel",
        "preload":None,
        "model_folder":"model_folder",
        "tokenizer_file":"tokenizer_{0}.json",
        "experiment_name":"runs/model"

    }

def get_weights_file_path(config,epochs):
    model_folder=config['model_folder']
    model_basename=config['model_basename']
    model_filename=f"{model_basename}{epochs}.pt"
    return str(Path('.')/model_folder/model_filename)
