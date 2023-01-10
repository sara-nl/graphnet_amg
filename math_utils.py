import numpy as np
import tensorflow as tf
import scipy
from scipy.sparse import csr_matrix

'''functions for math operations'''

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
                                        debug=None, check_finite=False )
    return S

# originally in matlab, now in python
def P_square_sparsity_pattern_py(P, size, coarse_nodes):
    size = int(size)
    P_coo = P.tocoo()
    P_rows, P_cols, P_values = P_coo.row, P_coo.col, P_coo.data
    P = csr_matrix((P_values, (P_rows, P_cols)), shape=(size, len(coarse_nodes)))
    P_square = csr_matrix((size, size))
    P_square[:, coarse_nodes] = P
    rows, cols = P_square.nonzero()

    return rows, cols



if __name__ == '__main__':
    P_baseline = None   

