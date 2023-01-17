import numpy as np
import tensorflow as tf
import fire
import os
import scipy
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import pyamg
import time

import h5py

import configs
from data import generate_A

'''Utility functions to save and load data matrices As to file'''
# TODO: make it better -> can create 1000 matrices in about 10 min, BUT out of memory for 5000 matrices

def save_to_file(As, data_config):
    num_As =  len(As)
    np_matrices = [mat.toarray() for mat in As]

    filename = f"data_dir/numAs_{num_As}_points_{data_config.num_unknowns}_blocks_{data_config.num_blocks}.h5"

    with h5py.File(filename, "w") as f:
        f.create_dataset("matrices", data=np_matrices)

def load_from_file(filename, run=0):
    if not os.path.isfile(filename):
        raise RuntimeError(f"file {filename} not found")
    with h5py.File(filename, "r") as f:
        # load the matrices from the dataset
        np_loaded = f["matrices"][()]
        As = [csr_matrix(mat) for mat in np_loaded]
 
    return As

# for testing or to run as a script
def generate_and_save_As(num_As, data_config):
    start_time = time.time()
    As = [generate_A(   data_config.num_unknowns,
                        data_config.num_blocks,
                        data_config.dist) for _ in range(num_As)]
    print("--- %s seconds for generating data ---" % (time.time() - start_time))

    start_time2 = time.time()
    save_to_file(As, data_config)
    print("--- %s seconds for saving data ---" % (time.time() - start_time2))

    return As

# utils for visualize distribution of the created dataset
def visualize_distribution(As):
    nonzero_values = [A.data for A in As]
    data = np.concatenate(nonzero_values)

    plt.hist(data, density=True, bins=30)  # density=False would make counts
    plt.ylabel('Frequency')
    plt.xlabel('Data')
    plt.savefig('created_As_histogram.png')


if __name__ == '__main__':

    config = getattr(configs, 'GRAPH_LAPLACIAN_TRAIN')
    data_config = config.data_config
    num_As = 1000
    As = generate_and_save_As(num_As, data_config)

    # for testing load file
    #filename = f"data_dir/numAs_{num_As}_points_{data_config.num_unknowns}_blocks_{data_config.num_blocks}.h5"
    #start_time = time.time()
    #As_loaded = load_from_file(filename)
    #print("--- %s seconds for generating data ---" % (time.time() - start_time))
    #visualize_distribution(As_loaded)

    






