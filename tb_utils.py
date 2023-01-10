import tensorflow as tf
import random
import string
import os
import shutil
import glob

"""
Utility functions to record logs in tensorboard
"""

# configure model name and where to save the logs
def set_log_model():
    model_name = ''.join(random.choices(string.digits, k=5))  # to make the model_name string unique
    create_results_dir(model_name)
    logdir = './tb_dir/' + model_name

    summary_writer = tf.summary.create_file_writer(logdir)

    return summary_writer

def create_results_dir(model_name):
    results_dir = 'results/' + model_name
    os.makedirs(results_dir)

    # make a copy of all Python files, for reproducibility
    local_dir = os.path.dirname(__file__)
    for py_file in glob.glob(local_dir + '/*.py'):
        shutil.copy(py_file, results_dir)

# to record log into tensorboard
def record_tb(summary_writer, frob_loss, frob_loss_pyamg):
    
    with summary_writer.as_default():   
        # TODO test if works outside, otherwise check this
            # https://stackoverflow.com/questions/56961856/how-to-write-to-tensorboard-in-tensorflow-2
        record_tb_loss(frob_loss, frob_loss_pyamg)
        summary_writer.flush()

    return None

def record_tb_loss(frob_loss, frob_loss_pyamg):
    with tf.name_scope("losses"):
        tf.summary.scalar("frob_loss", frob_loss, step=tf.compat.v1.train.get_global_step())
        tf.summary.scalar("frob_loss_pyamg", frob_loss_pyamg, step=tf.compat.v1.train.get_global_step())
