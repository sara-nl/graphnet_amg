import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import time

import configs
from data import generate_A, save_to_file, load_from_file

'''Utility function to save and load data matrices As to file'''

#### Utils for running this file as script
# for testing or to run as a script
def generate_and_save_As(num_As, data_config):
    filename = f"data_dir/numAs_{num_As}_points_{data_config.num_unknowns}_blocks_{data_config.num_blocks}.npy"

    start_time = time.time()
    As = [generate_A(   data_config.num_unknowns,
                        data_config.num_blocks,
                        data_config.dist) for _ in range(num_As)]
    print("--- %s seconds for generating data ---" % (time.time() - start_time))

    start_time2 = time.time()
    save_to_file(As, filename)
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
    num_As = 3000
    As = generate_and_save_As(num_As, data_config)

    # for testing load file
    #filename = f"data_dir/numAs_{num_As}_points_{data_config.num_unknowns}_blocks_{data_config.num_blocks}.npy"
    #start_time = time.time()
    #As_loaded = load_from_file(filename)
    #print("--- %s seconds for generating data ---" % (time.time() - start_time))
    #visualize_distribution(As_loaded)

    






