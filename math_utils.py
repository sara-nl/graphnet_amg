import numpy as np
import tensorflow as tf
import scipy
from scipy.sparse import csr_matrix

'''functions for math operations'''

def to_prolongation_matrix(P_square, P_baseline, coarse_nodes):
    # set the diagonal of P_square to 1
    # this is because an error/residual from a coarse node is itself after prolongation
    P_square.setdiag(np.ones(P_square.shape[0]))
    # remove all columns of P_square associated with fine nodes
    # note P is mxn for A of mxm and A_coarse of nxn
    P = P_square[:, coarse_nodes]
    # apply the sparsity pattern of P_base to P
    P_baseline_mask = (P_baseline != 0).astype(np.float64)
    P = P.multiply(P_baseline_mask)
    P.eliminate_zeros()
    # rescale rows of P to have the same row sums as P_base
    baseline_row_sum = P_baseline.sum(axis=1)
    baseline_row_sum = np.array(baseline_row_sum)[:, 0]
    P_row_sum = np.array(P.sum(axis=1))[:, 0]
    # https://stackoverflow.com/a/12238133
    P_copy = P.copy()
    P_copy.data /= P_row_sum.repeat(np.diff(P_copy.indptr))
    P_copy.data *= baseline_row_sum.repeat(np.diff(P_copy.indptr))
    P = P_copy
    return P


def to_prolongation_matrix_tensor(matrix, coarse_nodes, baseline_P, nodes,
                                  normalize_rows=True,
                                  normalize_rows_by_node=False):
    dtype = tf.float64
    matrix = tf.cast(matrix, dtype)
    matrix = tf.sparse.to_dense(matrix)

    # prolongation from coarse point to itself should be identity. This corresponds to 1's on the diagonal
    matrix = tf.linalg.set_diag(matrix, tf.ones(matrix.shape[0], dtype=dtype))

    # select only columns corresponding to coarse nodes
    matrix = tf.gather(matrix, coarse_nodes, axis=1)

    # set sparsity pattern (interpolatory sets) to be of baseline prolongation
    baseline_zero_mask = tf.cast(tf.not_equal(baseline_P, tf.zeros_like(baseline_P)), dtype)
    matrix = matrix * baseline_zero_mask

    if normalize_rows:
        if normalize_rows_by_node:
            baseline_row_sum = nodes
        else:
            baseline_row_sum = tf.reduce_sum(baseline_P, axis=1)
        baseline_row_sum = tf.cast(baseline_row_sum, dtype)

        matrix_row_sum = tf.reduce_sum(matrix, axis=1)
        matrix_row_sum = tf.cast(matrix_row_sum, dtype)

        # there might be a few rows that are all 0's - corresponding to fine points that are not connected to any
        # coarse point. We use "divide_no_nan" to let these rows remain 0's
        matrix = tf.math.divide_no_nan(matrix, tf.reshape(matrix_row_sum, (-1, 1)))

        matrix = matrix * tf.reshape(baseline_row_sum, (-1, 1))
    return matrix

def to_prolongation_matrix_csr(matrix, coarse_nodes, baseline_P, nodes, normalize_rows=True,
                               normalize_rows_by_node=False):
    """
    sparse version of the above function, inputs are csr matrices
    """
    # prolongation from coarse point to itself should be identity. This corresponds to 1's on the diagonal
    matrix.setdiag(np.ones(matrix.shape[0]))

    # select only columns corresponding to coarse nodes
    matrix = matrix[:, coarse_nodes]

    # set sparsity pattern (interpolatory sets) to be of baseline prolongation
    baseline_P_mask = (baseline_P != 0).astype(np.float64)
    matrix = matrix.multiply(baseline_P_mask)
    matrix.eliminate_zeros()

    if normalize_rows:
        if normalize_rows_by_node:
            baseline_row_sum = nodes
        else:
            baseline_row_sum = baseline_P.sum(axis=1)
            baseline_row_sum = np.array(baseline_row_sum)[:, 0]

        matrix_row_sum = np.array(matrix.sum(axis=1))[:, 0]
        # https://stackoverflow.com/a/12238133
        matrix_copy = matrix.copy()
        matrix_copy.data /= matrix_row_sum.repeat(np.diff(matrix_copy.indptr))
        matrix_copy.data *= baseline_row_sum.repeat(np.diff(matrix_copy.indptr))
        matrix = matrix_copy
    return matrix


def P_square_sparsity_pattern(P_csr, n_node, coarse_nodes):
    P_coo = P_csr.tocoo()

    P_square = np.zeros((n_node, n_node))
    i_coo = 0  # index of coo item
    for i_row in P_coo.row:
        i_col = P_coo.col[i_coo]
        i_col_square = coarse_nodes[i_col]
        P_square[i_row][i_col_square] = P_coo.data[i_coo]
        i_coo += 1

    return np.nonzero(P_square)


"""
# The following is Luz's implementation. It takes a batch of matrices as input
# in this code power=1 means squared Frobenius norm
def frob_norm(m, power=1):
    if power == 1:
        return tf.norm(m, axis=[-2,-1])
    else:
        curr_power = m
        for i in range(power - 1):
            curr_power = m @ curr_power
        return tf.norm(curr_power, axis=[-2,-1]) ** (1/power)
"""


def frob_norm(m):
    """
    :param m: a matrix as tf.Tensor
    :return: the 2nd order Frobenius norm
    """
    return tf.math.reduce_sum(m ** 2) ** (1 / 2)


def two_grid_error_matrix(A, P, S):
    """
    given A, P, and S as tf.Tensor
    :param A:
    :param P:
    :param S:
    :return: error matrix M as tf.Tensor
    """
    C = compute_C(A, P)
    M = S @ C @ S
    return M


def compute_C(A, P):
    """
    given A amd P as tf.Tensor
    :param A:
    :param P:
    :return: C = I - P * A_coarse_inv * R * A (C is a measure of error propagation by going down to a coarser grid)
    """
    R = tf.transpose(P)
    A_coarse = R @ A @ P
    RA = R @ A
    A_coarse_inv = tf.linalg.inv(A_coarse)
    A_coarse_inv_RA = A_coarse_inv @ RA
    P_A_coarse_inv_RA = P @ A_coarse_inv_RA
    I = tf.eye(len(A), dtype=tf.float64)
    C = I - P_A_coarse_inv_RA
    return C

def csr_to_tensor(csr_list):
    return
def decompose_matrix(A):
  n, m = A.shape
  L = np.tril(A, k=-1)
  D = np.diag(np.diagonal(A))
  U = np.triu(A, k=1)
  return L, D, U


# new implementation instead of relaxation.py
def compute_relaxation_matrices(A, tensor=False):
    A = A.toarray()
    L, D, U = decompose_matrix(A)
    a = D + L
    S = scipy.linalg.solve_triangular(  a=a, b= -U, 
                                        lower=True, unit_diagonal=False, overwrite_b=True, 
                                        check_finite=False )
    return S


if __name__ == '__main__':
    P_baseline = None   