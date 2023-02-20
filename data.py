import math
from functools import lru_cache
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

"""
Collection of method to generate/save/load the dataset
supports loading of As per run to reduce the memory overload when num_As is big
"""


def generate_A(points, num_blocks, dist="sparse_block_circulant"):
    """
    Input: num of unknowns, num of blocks, distribution
    Output: a matrix A in csr format
    """
    if dist is "sparse_block_circulant":
        # custom generated BC matrices, SPD BC if the flag is set to true
        A = generate_doubly_block_circulant(
            points, num_blocks, sparsity=0.01, flag_SPD=True
        )
    elif dist is "poisson":
        grid_size = int(np.sqrt(points))
        A = pyamg.gallery.poisson((grid_size, grid_size), type="FE")
    elif dist is "aniso":
        grid_size = int(np.sqrt(points))
        stencil = pyamg.gallery.diffusion_stencil_2d(
            epsilon=0.01, theta=np.pi / 3, type="FE"
        )
        A = pyamg.gallery.stencil_grid(stencil, (grid_size, grid_size), format="csr")
    elif dist is "example":
        A = pyamg.gallery.load_example(points)["A"]
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
        c_blocks = [
            scipy.sparse.random(
                c, c, density=sparsity / ((i + 1) ** 3), data_rvs=rvs
            ).toarray()
            for i in range(b)
        ]

        block = block_circulant(c_blocks)
        B.append(block)

    # input B is a tuple of arrays corresponding to the first block column of the output block matrix
    A = block_circulant(B)
    if flag_SPD:
        A = make_it_SPD(A)

    return csr_matrix(A)


def create_dataset(numAs, data_config, run=0, eval = False):
    """
    At every run load or generate training samples As and return the DataSet { As, Ss, coarse list, baseline P }

    --> how to use it in the main function:
    "samples_per_run = config.data_config.num_As // config.train_config.num_runs

    for run in range(config.train_config.num_runs):
        run_dataset = create_dataset(samples_per_run, config.data_config, run)
        train(run_dataset)

    """
    As_filename = \
        f"{data_config.data_dir}numAs_{numAs}_points_{data_config.num_unknowns}_blocks_{data_config.num_blocks}"
    if not eval:
        As_filename = As_filename + f"_run_{run}"
    As_filename = As_filename + f".npy"
    if os.path.exists(As_filename):
        print("file exists, loading As from file")
        As = load_from_file(As_filename)
    else:
        print("generating new As")
        As = generate_in_one_file(numAs, data_config, As_filename, run)

    return create_dataset_from_As(As, data_config)


def create_dataset_from_As(As, data_config):
    """
    return DataSet which contains
        As: list of csr
        Ss: list of numpy arrays
        coarse_node_list: list of numpy arrays
        baseline_P_list: list of Tensorflow.Tensor objects
    """
    num_As = len(As)
    print("As loaded, now creating Ss")

    # Ss = [None] * num_As  # relaxation matrices are only created per block when calling loss()
    Ss = [compute_relaxation_matrices(A) for A in As]

    solvers = [
        pyamg.ruge_stuben_solver(A, max_levels=2, keep=True, CF=data_config.splitting)
        for A in As
    ]
    baseline_P_list = [solver.levels[0].P for solver in solvers]
    # baseline_P_list = [tf.convert_to_tensor(P.toarray(), dtype=tf.float64) for P in baseline_P_list]
    splittings = [solver.levels[0].splitting for solver in solvers]
    coarse_nodes_list = [np.nonzero(splitting)[0] for splitting in splittings]

    return DataSet(As, Ss, coarse_nodes_list, baseline_P_list)


# utils for load and save to file
def generate_in_one_file(samples_per_run, data_config, filename, run=0):
    """
    Utility function that generates As and save them in one file
    Returns As
    """
    # filename = f"{data_config.data_dir}numAs_{samples_per_run}_points_{data_config.num_unknowns}_blocks_{data_config.num_blocks}_{run}.npy"
    As = [
        generate_A(data_config.num_unknowns, data_config.num_blocks, data_config.dist)
        for _ in range(samples_per_run)
    ]
    if data_config.save_data:
        save_to_file(As, filename)

    return As


def save_to_file(As, filename):
    np_matrices = np.array(As)
    np.save(filename, np_matrices, allow_pickle=True)


def load_from_file(filename, run=0):
    """
    Utility to load As from a single file
    """
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
        print("Warning: matrix not symmetric")
        return False


def make_it_SPD(matrix):
    if not is_symmetric(matrix):
        matrix = (matrix + matrix.T) / 2

    if not is_positive_definite(matrix):
        eigenvalues = np.linalg.eigvals(matrix)
        min_eigenvalue = np.min(eigenvalues)
        matrix += (np.abs(min_eigenvalue) + 1) * np.eye(matrix.shape[0])

    return matrix

### Functions used for testing:

def As_poisson_grid(num_As, num_unknowns, constant_coefficients=False):
    """
    Generate a list of sparse matrices representing Poisson operators on a square grid.

    num_As: Number of matrices to generate.
    num_unknowns: Number of unknowns in each matrix. Should be a perfect square.
    constant_coefficients: If True, use constant diffusion coefficients to generate the stencils. If False,
                                randomly generate the diffusion coefficients.
    return: A list of sparse matrices representing Poisson operators on a square grid.

    :raises RuntimeError: If num_unknowns is not a perfect square.

    The matrices are computed using the stencils generated by the `poisson_dirichlet_stencils` function.
    """
    grid_size = int(math.sqrt(num_unknowns))
    if grid_size ** 2 != num_unknowns:
        raise RuntimeError("num_unknowns must be a square number")
    stencils = poisson_dirichlet_stencils(num_As, grid_size, constant_coefficients=constant_coefficients)
    A_idx, stencil_idx = compute_A_indices(grid_size)
    matrices = []
    for stencil in stencils:
        matrix = csr_matrix(arg1=(stencil.reshape((-1)), (A_idx[:, 0], A_idx[:, 1])),
                            shape=(grid_size ** 2, grid_size ** 2))
        matrix.eliminate_zeros()
        matrices.append(matrix)
    return matrices

def poisson_dirichlet_stencils(num_stencils, grid_size, constant_coefficients=False):
    '''
    Generate a set of stencils for solving the Poisson equation with Dirichlet boundary conditions on a regular 2D grid. 
    The stencils are defined using finite difference approximations of the Laplacian operator.
    '''
    stencil = np.zeros((num_stencils, grid_size, grid_size, 3, 3))

    if constant_coefficients:
        diffusion_coeff = np.ones(shape=[num_stencils, grid_size, grid_size])
    else:
        diffusion_coeff = np.exp(np.random.normal(size=[num_stencils, grid_size, grid_size]))

    jm1 = [(i - 1) % grid_size for i in range(grid_size)]
    stencil[:, :, :, 1, 2] = -1. / 6 * (diffusion_coeff[:, jm1] + diffusion_coeff)
    stencil[:, :, :, 2, 1] = -1. / 6 * (diffusion_coeff + diffusion_coeff[:, :, jm1])
    stencil[:, :, :, 2, 0] = -1. / 3 * diffusion_coeff[:, :, jm1]
    stencil[:, :, :, 2, 2] = -1. / 3 * diffusion_coeff

    jp1 = [(i + 1) % grid_size for i in range(grid_size)]

    stencil[:, :, :, 1, 0] = stencil[:, :, jm1, 1, 2]
    stencil[:, :, :, 0, 0] = stencil[:, jm1][:, :, jm1][:, :, :, 2, 2]
    stencil[:, :, :, 0, 1] = stencil[:, jm1][:, :, :, 2, 1]
    stencil[:, :, :, 0, 2] = stencil[:, jm1][:, :, jp1][:, :, :, 2, 0]
    stencil[:, :, :, 1, 1] = -np.sum(np.sum(stencil, axis=4), axis=3)

    stencil[:, :, 0, :, 0] = 0.
    stencil[:, :, -1, :, -1] = 0.
    stencil[:, 0, :, 0, :] = 0.
    stencil[:, -1, :, -1, :] = 0.
    return stencil

@lru_cache(maxsize=None)
def compute_A_indices(grid_size):
    K = map_2_to_1(grid_size=grid_size)
    A_idx = []
    stencil_idx = []
    for i in range(grid_size):
        for j in range(grid_size):
            I = int(K[i, j, 1, 1])
            for k in range(3):
                for m in range(3):
                    J = int(K[i, j, k, m])
                    A_idx.append([I, J])
                    stencil_idx.append([i, j, k, m])
    return np.array(A_idx), stencil_idx

def map_2_to_1(grid_size):
    # maps 2D coordinates to the corresponding 1D coordinate in the matrix.
    k = np.zeros((grid_size, grid_size, 3, 3))
    M = np.reshape(np.arange(grid_size ** 2), (grid_size, grid_size)).T
    M = np.concatenate([M, M], 0)
    M = np.concatenate([M, M], 1)
    for i in range(3):
        I = (i - 1) % grid_size
        for j in range(3):
            J = (j - 1) % grid_size
            k[:, :, i, j] = M[I:I + grid_size, J:J + grid_size]
    return k


if __name__ == "__main__":
    # for testing
    # np.random.seed(1)
    A = generate_doubly_block_circulant(c=5, b=3, sparsity=0.2, flag_SPD=True)

    print(is_positive_definite(A.toarray()))
    print(np.linalg.eigvals(A.toarray()))