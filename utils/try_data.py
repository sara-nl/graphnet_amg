import math
import matplotlib.pyplot as plt
from collections import defaultdict
import tensorflow as tf
import matlab
import meshpy.triangle as triangle
import numpy as np
import pyamg
import scipy
from scipy.sparse import csr_matrix
import configs
import matlab.engine
import numpy as np
import pandas as pd

from data import generate_doubly_block_circulant, is_positive_definite, is_symmetric

# from matlab
def generate_A_delaunay_block_periodic_lognormal(num_unknowns_per_block, root_num_blocks, matlab_engine):
    """Poisson equation on triangular mesh, with lognormal coefficients, and block periodic boundary conditions"""
    # points are correct only for 3x3 number of blocks
    A_matlab, points_matlab = matlab_engine.block_periodic_delaunay(num_unknowns_per_block, root_num_blocks, nargout=2)
    A_numpy = np.array(A_matlab._data).reshape(A_matlab.size, order='F')
    points_numpy = np.array(points_matlab._data).reshape(points_matlab.size, order='F')
    return csr_matrix(A_numpy)


# for testing
def is_doubly_block_circulant(A, c, b):
    # TODO add documentation https://www.scirp.org/pdf/apm_2017020914093511.pdf
    """   
    Verify if the given matrix A is doubly block circulant.
    Steps:
    - Construct a shift matrix T of dim (b x b)
    - A is doubly block circulant if H * A = A * H 
        where H = kron product of eye(b), T and eye(c)
    - A is block circulant if S * A = A * S
        where S = kron product of T and eye(b*c)

    Inputs:
    - A: matrix to test
    - c: num of rows of the smallest blocks
    - b: num of the blocks.
    
    Returns:
    True/False
    """
    Ib, Ic, Ibc = np.eye(b), np.eye(c), np.eye(b*c)

    T = np.concatenate((Ib[1:b, :], Ib[0:1, :]), axis=0)
    H = np.kron(Ib, np.kron(T,Ic))
    S = np.kron(T, Ibc)

    #check inner blocks 
    if  np.allclose(H*A, A*H):
        # check also externally
        return np.allclose(S*A, A*S) 
    else:
        print('WARNING: The inner blocks are not circulant')
        return False

      
def visualize_distribution(arrs, legs , filename = 'histogram.png'):
    n_arrs = len(arrs) 
    fig, axs = plt.subplots(1, n_arrs) 

    for i, (arr, leg) in enumerate(zip(arrs, legs)):
        values = arr.data
        axs[i].hist(values)
        axs[i].set_title(leg)
        axs[i].set_xlabel("Value")
        axs[i].set_ylabel("Density")

    plt.savefig(filename)



def compare_data():
    matlab_engine = matlab.engine.start_matlab()

    #make this example reproducible
    seed = 1
    np.random.seed(seed)
    tf.random.set_random_seed(seed)
    matlab_engine.eval(f'rng({seed})')

    num_unknowns = 64
    num_blocks= 4
    
    # num_rows = b^2 c

    # from luz's
    A = generate_A_delaunay_block_periodic_lognormal(num_unknowns, num_blocks, matlab_engine)

    print(is_positive_definite(A.toarray()))
    #print(np.linalg.eigvals(A.toarray()))
    print(is_doubly_block_circulant(A.toarray(), num_unknowns, num_blocks))

    B = generate_doubly_block_circulant(num_unknowns, num_blocks, sparsity=0.01, flag_SPD = False)
    C = generate_doubly_block_circulant(num_unknowns, num_blocks, sparsity=0.01, flag_SPD = True)

    print(is_positive_definite(C.toarray()))
    print(is_doubly_block_circulant(C.toarray(), num_unknowns, num_blocks))

    arrs = [A, B, C]
    legs = ['luzs', 'DBC', 'SPD DBC']
    visualize_distribution(arrs, legs, 'dist_comparison.png')


if __name__ == '__main__':
    compare_data()
