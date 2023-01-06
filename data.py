import numpy as np
import pyamg
import scipy
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.linalg import circulant
from sklearn.datasets import make_sparse_spd_matrix

from sporco.linalg import block_circulant


def generate_A(dist, points, num_blocks):
    if dist is 'sparse_block_circulant':
        # custom generated BC matrices, SPD BC if the flag is set to true
        A = generate_doubly_block_circulant(points, num_blocks, sparsity=0.01, flag_SPD = True)
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
    #rvs = stats.poisson(25, loc=10).rvs
    
    # doubly block circulant: first generate b small blocks of size c and then bigger b blocks
    B = []
    for _ in range(b):
        c_blocks = [scipy.sparse.random(c, c, density=sparsity/((i+1) ** 3), data_rvs=rvs).toarray() for i in range(b)] 

        block = block_circulant(c_blocks)
        B.append(block)
    
    # input B is a tuple of arrays corresponding to the first block column of the output block matrix
    A = block_circulant(B)
    if flag_SPD:
        A = make_it_SPD(A)

    return csr_matrix(A)


# utils
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
    np.random.seed(1)
    A =  generate_doubly_block_circulant(points = 5, num_blocks = 3, sparsity =0.2, flag_SPD= True)

    print(is_positive_definite(A.toarray()))
    print(np.linalg.eigvals(A.toarray()))
