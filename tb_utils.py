import tensorflow as tf
import random
import string
import os
import shutil
import json
import glob
import numpy as np

from tensorflow.python.client import device_lib

"""
Utility functions to set TF2 and record logs in tensorboard
"""

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos] # example of output => ['/device:CPU:0', '/device:GPU:0']


def config_tf():
    """
    Set TF2 (which is by default in eager execution) to support gpu memory growth
    """
    os.environ["TF_ENABLE_TENSORRT"] = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] ="0" 
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                print("Name:", gpu.name, "  Type:", gpu.device_type)
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

    record_params_every = max(300 // batch_size, 1)
    if batch % record_params_every == 0:
        record_tb_params(iteration, batch_size, grads, variables)


def record_tb_eval(M, run, num_As, iteration, batch, batch_size, eval_loss, eval_M):
    batch = run * num_As + batch

    record_spectral_every = max(300 // batch_size, 1)
    if batch % record_spectral_every == 0:
        record_tb_spectral_radius(iteration, M, eval_loss, eval_M)


def record_tb_spectral_radius(iteration, M, eval_loss, eval_M):

    spectral_radius = np.abs(np.linalg.eigvals(M.numpy())).max()
    eval_spectral_radius = np.abs(np.linalg.eigvals(eval_M.numpy())).max()

    tf.summary.scalar('spectral_radius', spectral_radius, step=iteration)
    tf.summary.scalar('eval_loss', eval_loss, step = iteration)
    tf.summary.scalar('eval_spectral_radius', eval_spectral_radius, step=iteration)


        
def record_tb_params(iteration, batch_size, grads, variables):
   
    for i in range(len(variables)):
        variable = variables[i]
        variable_name = variable.name
        grad = grads[i]
        if grad is not None:
            tf.summary.scalar(variable_name + '_grad', tf.norm(grad) / batch_size, step=iteration)
            tf.summary.histogram(variable_name + '_grad_histogram', grad / batch_size, step=iteration)
            tf.summary.scalar(variable_name + '_grad_fraction_dead', tf.nn.zero_fraction(grad), step=iteration)
            tf.summary.scalar(variable_name + '_value', tf.norm(variable), step=iteration)
            tf.summary.histogram(variable_name + '_value_histogram', variable, step=iteration)


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
