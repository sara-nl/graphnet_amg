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

def record_tb(summary_writer, frob_loss, frob_loss_pyamg=None):
    """to record log into tensorboard"""

    with summary_writer.as_default():
        # TODO to test it during training, otherwise check this
        # https://stackoverflow.com/questions/56961856/how-to-write-to-tensorboard-in-tensorflow-2
        record_tb_loss(frob_loss, frob_loss_pyamg)
        summary_writer.flush()

    return None


def record_tb_loss(frob_loss, frob_loss_pyamg=None):
    with tf.name_scope("losses"):
        tf.summary.scalar(
            "frob_loss", frob_loss, step=tf.compat.v1.train.get_global_step()
        )
        if frob_loss_pyamg != None:
            tf.summary.scalar(
                "frob_loss_pyamg",
                frob_loss_pyamg,
                step=tf.compat.v1.train.get_global_step(),
            )


def write_config_file(run_name, config):
    results_dir = "results/" + run_name
    config_dict = {"train_config": config.train_config.__dict__,
                   "data_config": config.data_config.__dict__,
                   "model_config": config.model_config.__dict__,
                   "run_config": config.run_config.__dict__}
    with open(f"{results_dir}/configs.json","w") as outfile:
        json.dump(config_dict, outfile)