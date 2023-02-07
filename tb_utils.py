import tensorflow as tf
import random
import string
import os
import shutil
import json
import glob

"""
Utility functions to set TF2 and record logs in tensorboard
"""


def config_tf():
    """
    Set TF2 (which is by default in eager execution) to support gpu memory growth
    """
    os.environ["TF_ENABLE_TENSORRT"] = "0"
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def set_log_model():
    """configure model name and where to save the logs"""

    model_name = "".join(
        random.choices(string.digits, k=5)
    )  # to make the model_name string unique
    create_results_dir(model_name)
    logdir = "./tb_dir/" + model_name

    summary_writer = tf.summary.create_file_writer(logdir)

    return summary_writer


def create_results_dir(model_name):
    results_dir = "results/" + model_name
    os.makedirs(results_dir)

    # make a copy of all Python files, for reproducibility
    local_dir = os.path.dirname(__file__)
    for py_file in glob.glob(local_dir + "/*.py"):
        shutil.copy(py_file, results_dir)
    return


def record_tb(M, run, num_As, iteration, batch, batch_size, frob_loss, grads, loop, model, variables, eval_dataset,
              eval_A_graphs_tuple, eval_config):
    batch = run * num_As + batch
    record_loss_every = max(1 // batch_size, 1)
    
    if batch % record_loss_every == 0:
        record_tb_loss(iteration, frob_loss)

    #record_params_every = max(300 // batch_size, 1)
    #if batch % record_params_every == 0:
    #    record_tb_params(batch_size, grads, loop, variables)

    #record_spectral_every = max(300 // batch_size, 1)
    #if batch % record_spectral_every == 0:
    #    record_tb_spectral_radius(M, model, eval_dataset, eval_A_graphs_tuple, eval_config)

        


def record_tb_loss(iteration, frob_loss):
    tf.summary.scalar('loss', frob_loss, step=tf.keras.backend.get_value(iteration))


def write_config_file(run_name, config):
    results_dir = "results/" + run_name
    config_dict = {
        "train_config": config.train_config.__dict__,
        "data_config": config.data_config.__dict__,
        "model_config": config.model_config.__dict__,
        # "run_config": config.run_config.__dict__
    }
    with open(f"{results_dir}/configs.json", "w") as outfile:
        json.dump(config_dict, outfile)
