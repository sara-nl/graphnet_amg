import numpy as np
import tensorflow as tf
import os
import scipy
from scipy.sparse import csr_matrix
import pyamg

from data import generate_A
from dataset import DataSet
from math_utils import compute_relaxation_matrices

'''Utility functions to construct the class Dataset based on the data'''

def create_dataset(num_As, data_config, run=0):
    # load As from file or generate new ones
    As_filename = f"data_dir/training_As.npy"
    if os.path.exists(As_filename):
        load_from_file(As_filename, run)
    else:
        As = [generate_A(data_config.num_unknowns,
                         data_config.num_blocks,
                         data_config.dist) for _ in range(num_As)]

    return create_dataset_from_As(As, data_config)

def create_dataset_from_As(As, data_config):
    num_As = len(As)
    
    #Ss = [None] * num_As  # relaxation matrices are only created per block when calling loss()
    Ss = [compute_relaxation_matrices(A) for A in range(num_As)]

    solvers = [pyamg.ruge_stuben_solver(A, max_levels=2, keep=True, CF=data_config.splitting)
                   for A in As]
    baseline_P_list = [solver.levels[0].P for solver in solvers]
    baseline_P_list = [tf.convert_to_tensor(P.toarray(), dtype=tf.float64) for P in baseline_P_list]
    splittings = [solver.levels[0].splitting for solver in solvers]
    coarse_nodes_list = [np.nonzero(splitting)[0] for splitting in splittings]
    
    return DataSet(As, Ss, coarse_nodes_list, baseline_P_list)


# TODO load data from existing file: 
def load_from_file(As_filename, run=0):
    # load data based on index run
    if not os.path.isfile(As_filename):
        raise RuntimeError(f"file {As_filename} not found")
    As = np.load(As_filename, allow_pickle=True)

    # workaround for data generated with both matrices and point coordinates
    if len(As.shape) == 1:
        As = list(As)
    elif len(As.shape) == 2:
        As = list(As[0])
    
    return As





if __name__ == '__main__':
    P_baseline = None   

