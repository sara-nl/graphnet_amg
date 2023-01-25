import numpy as np
import os
import tensorflow as tf
import pyamg
import scipy
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.linalg import circulant

from sporco.linalg import block_circulant

from dataset import DataSet
from math_utils import compute_relaxation_matrices


def generate_A(points, num_blocks, dist='sparse_block_circulant'):
    if dist is 'sparse_block_circulant':
        # custom generated BC matrices, SPD BC if the flag is set to true
        A = generate_doubly_block_circulant(points, num_blocks, sparsity=0.01, flag_SPD=True)
    elif dist is 'poisson':
        grid_size = int(np.sqrt(points))
        A = pyamg.gallery.poisson((grid_size, grid_size), type='FE')
    elif dist is 'aniso':
        grid_size = int(np.sqrt(points))
        stencil = pyamg.gallery.diffusion_stencil_2d(epsilon=0.01, theta=np.pi / 3, type='FE')
        A = pyamg.gallery.stencil_grid(stencil, (grid_size, grid_size), format='csr')
    elif dist is 'example':
        A = pyamg.gallery.load_example(points)['A']
    return A


def generate_doubly_block_circulant(c, b, sparsity, flag_SPD):
    """
    Generate a sparse doubly block circulant matrix with coefficients. from lognorm distribution
    A is symmetric and positive definite.

    Inputs:
    - c: num of rows of the smallest blocks
    - b: num of the blocks
    - sparsity: density of the generated matrix, 1: a full matrix, 0: no non-zero items
    - flag_SPD: flag to impose SPD property on the generated A.
    
    Returns:
    A sparse block circulant matrix of size 'n = b^2 * c' with b blocks of size 'k = c * b'.
    """
    # k = c * b
    # n = k * b

    # set random values from standard lognormal distribution mu = 0 sigma = 1
    rvs = stats.lognorm(s=1).rvs
    # rvs = stats.poisson(25, loc=10).rvs

    # doubly block circulant: first generate b small blocks of size c and then bigger b blocks
    B = []
    for _ in range(b):
        c_blocks = [scipy.sparse.random(c, c, density=sparsity / ((i + 1) ** 3), data_rvs=rvs).toarray() for i in
                    range(b)]

        block = block_circulant(c_blocks)
        B.append(block)

    # input B is a tuple of arrays corresponding to the first block column of the output block matrix
    A = block_circulant(B)
    if flag_SPD:
        A = make_it_SPD(A)

    return csr_matrix(A)


# generate dataset
def create_dataset(data_config, run=0):
    # load As from file or generate new ones
    As_filename = f"{0}numAs_{1}_points_{data_config.num_unknowns}_blocks_{data_config.num_blocks}.npy".format(
        data_config.datadir, data_config.num_As)
    if os.path.exists(As_filename):
        As = load_from_file(As_filename)
    else:
        As = [generate_A(data_config.num_unknowns,
                         data_config.num_blocks,
                         data_config.dist) for _ in range(data_config.num_As)]
        if data_config.save_data:
            save_to_file(As, As_filename)

    return create_dataset_from_As(As, data_config)


def create_dataset_from_As(As, data_config):
    num_As = len(As)

    # Ss = [None] * num_As  # relaxation matrices are only created per block when calling loss()
    Ss = [compute_relaxation_matrices(A) for A in As]

    solvers = [pyamg.ruge_stuben_solver(A, max_levels=2, keep=True, CF=data_config.splitting)
               for A in As]
    baseline_P_list = [solver.levels[0].P for solver in solvers]
    baseline_P_list = [tf.convert_to_tensor(P.toarray(), dtype=tf.float64) for P in baseline_P_list]
    splittings = [solver.levels[0].splitting for solver in solvers]
    coarse_nodes_list = [np.nonzero(splitting)[0] for splitting in splittings]

    return DataSet(As, Ss, coarse_nodes_list, baseline_P_list)


# utils for load and save to file
def save_to_file(As, filename):
    num_As = len(As)
    np_matrices = np.array(As)

    np.save(filename, np_matrices, allow_pickle=True)


def load_from_file(filename, run=0):
    if not os.path.isfile(filename):
        raise RuntimeError(f"file {filename} not found")
    np_loaded = np.load(filename, allow_pickle=True)
    As = [csr_matrix(mat) for mat in np_loaded]
    return As


# utils for matrix operations
def is_symmetric(A):
    return np.allclose(A, A.T)


def is_positive_definite(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        print('Warning: matrix not symmetric')
        return False


def make_it_SPD(matrix):
    if not is_symmetric(matrix):
        matrix = (matrix + matrix.T) / 2

    if not is_positive_definite(matrix):
        eigenvalues = np.linalg.eigvals(matrix)
        min_eigenvalue = np.min(eigenvalues)
        matrix += (np.abs(min_eigenvalue) + 1) * np.eye(matrix.shape[0])

    return matrix


if __name__ == '__main__':
    # for testing
    # np.random.seed(1)
    A = generate_doubly_block_circulant(c=5, b=3, sparsity=0.2, flag_SPD=True)

    print(is_positive_definite(A.toarray()))
    print(np.linalg.eigvals(A.toarray()))