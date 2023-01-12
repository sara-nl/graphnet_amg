import numpy as np
import tensorflow as tf


# Pre:
# Post:
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


# Pre:
# Post:
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
#The following is Luz's implementation. It takes a batch of matrices as input
# in this code power=1 means squared Frobenius norm
# Pre:
# Post:
def frob_norm(m, power=1):
    if power == 1:
        return tf.norm(m, axis=[-2,-1])
    else:
        curr_power = m
        for i in range(power - 1):
            curr_power = m @ curr_power
        return tf.norm(curr_power, axis=[-2,-1]) ** (1/power)
"""


# Pre: a matrix as tf.Tensor
# Post: the 2nd order Frobenius norm
def frob_norm(m):
    return tf.math.reduce_sum(m ** 2) ** (1 / 2)


# Pre: given A, P, and S as tf.Tensor
# Post: return error matrix M as tf.Tensor
def two_grid_error_matrix(A, P, S):
    C = compute_C(A, P)
    M = S @ C @ S
    return M


# Pre: given A amd P as tf.Tensor
# Post: return C = I - P * A_coarse_inv * R * A
#		C is a measure of error propagation by going down to a coarser grid
def compute_C(A, P):
    R = tf.transpose(P)
    A_coarse = R @ A @ P
    RA = R @ A
    A_coarse_inv = tf.linalg.inv(A_coarse)
    A_coarse_inv_RA = A_coarse_inv @ RA
    P_A_coarse_inv_RA = P @ A_coarse_inv_RA
    I = tf.eye(len(A))
    C = I - P_A_coarse_inv_RA
    return C

def csr_to_tensor(csr_list):
    return