import math
import pytest
import numpy as np
from scipy import sparse
import tensorflow as tf
import math_utils

matrix_A = tf.constant([[1., 7., 3.], [2., 1., 6.], [0., 4., 1.]])

matrix_P = tf.constant([[1., 2.], [0., 5.], [7., 1.]])

matrix_S = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])

matrix_C = tf.constant([[0.9762709, -0.36261493, -0.1394673], \
                        [0.07608497, -0.02826011, -0.01086928], \
                        [-0.36392447, 0.13517192, 0.05198902]])

matrix_M = tf.constant([[-0.05448174, -0.0366725, -0.01886326], \
                        [-3.1229234, -2.1019719, -1.0810204], \
                        [-6.1913652, -4.167271, -2.143177]])


def to_tensor(array):
    return tf.convert_to_tensor(array)


# def test_to_prolongation_matrix():


def test_matrix_mult():
    expected = tf.constant([[22, 40], [44, 15], [7, 21]])
    assert tf.experimental.numpy.allclose((matrix_A @ matrix_P), expected)


def test_frob_norm():
    computed = math_utils.frob_norm(matrix_M)
    expected = to_tensor(8.696929)
    assert math.isclose(computed, expected)


def test_two_grid_error_matrix():
    computed = math_utils.two_grid_error_matrix(matrix_A, matrix_P, matrix_S)
    expected = matrix_M()
    assert tf.experimental.numpy.allclose(computed, expected)


def test_compute_C():
    computed = math_utils.compute_C(matrix_A, matrix_P)
    expected = matrix_C()
    assert tf.experimental.numpy.allclose(computed, expected)


def test_p_square_sparsity_pattern():
    n_node = 4
    P = np.array([[1, 2],
                  [0, 1],
                  [0, 0],
                  [7, 1]
                  ])

    P_csr = sparse.csr_matrix(P)
    P_coo = P_csr.tocoo()
    coarse_nodes = [0, 2]

    computed = math_utils.P_square_sparsity_pattern(P_csr, n_node, coarse_nodes)
    expected = np.array([0, 0, 1, 3, 3]), np.array([0, 2, 2, 0, 2])
    # assert utils.P_square_sparsity_pattern(P_csr, n_node, coarse_nodes) == expected
    assert tf.experimental.numpy.allclose(computed, expected)