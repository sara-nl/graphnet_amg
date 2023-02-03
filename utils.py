import json
import os
import glob
import shutil

def create_results_dir(run_name):
    results_dir = "results/" + run_name
    os.makedirs(results_dir)

    # make a copy of all Python files
    local_dir = os.path.dirname(__file__)
    for py_file in glob.glob(local_dir = "/*.py"):
        shutil.copy(py_file, results_dir)
    return

def write_config_file(run_name, config):
    results_dir = "results/" + run_name
    config_dict = {"train_config": config.train_config.__dict__,
                   "data_config": config.data_config.__dict__,
                   "model_config": config.model_config.__dict__,
                   "run_config": config.run_config.__dict__}
    with open(f"{results_dir}/configs.json","w") as outfile:
        json.dump(config_dict, outfile)