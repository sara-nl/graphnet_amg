# graphnet_amg
Graph net boosted prolongation operator for algebraic multigrid

This code trains a graph neural network to select the interpolation weights in the prolongation operator for transporting errors between grids in the algebraic multigrid (AMG) algorithm.

AMG is an iterative algorithm for solving large and sparse linear systems that arise from, for example, discretisation of partial differential equations (PDEs). In each iteration, high frequency errors are eliminated by a classic relaxation scheme (commonly Gauss-Seidel), whereas low-frequency errors (i.e. smooth errors) are transferred onto a coarser grid where the smooth errors can be more efficiently removed by further relaxation sweeps. Multiple levels of such grids of increasing coarseness can be used for reducing the error. The interpolation operators transport errors between adjacent grid levels.

The coefficient matrix of a coarse grid, A_coarse, is commonly generated by the standard Galerkin projection: A_coarse = R * A * P = P_transpose * A * P, where the restriction matrix R (fine->coarse grid) is just the transpose of the prolongation matrix P (coarse->fine grid). P and R are the interpolation operators.

In particular, we use a neural network to help build the prolongation operator P that interpolates errors from a coarse grid to a fine one. This is done with the information of the coefficient matrix A of the finest grid, and the standard prolongation operator P_baseline generated by a traditional method. Although the sparsity pattern of P is enforced to be the same as P_baseline, the non-zero values (interpolation weights) in P are optimised through the neural network. Let matrix M describe the error propagation from one iteration to another by e_(i+1) = M * e_i, then the goal of an optimal P is to minimise M. In AMG, M = S_s1 * C * S_s2. S_s1 and S_s2 are the relaxation operators applied on the finest grid, before and after the cycle through the coarse grids, where s1 and s2 refer to the number of relaxation sweeps and are both typically 1 or 2. C represents the error reduction achieved through the coarse grids. C = I − P * [P_transpose * A * P]^(−1) * P_transpose * A. Given S_s1 and S_s2, which themselves typically do not have much room to be optimised, the choice of P determines the efficiency of error reduction per iteration. We use the Frobenius norm of M as the loss function of P. 

The objective of the neural network can be summarised as follows: given A and P_baseline, train P to minimise M. A is a matrix of size mxm, where m is the dimension of the unknowns in the linear system, i.e. the number of nodes in the finest grid. P_baseline is a matrix of size mxn, where n is the number of nodes in the coarser grid. n<m and typically n~m/2. P takes the same dimension and sparsity pattern of P_baseline. The problem can be naturally represented by a graph neural network where A is the adjacency matrix. The elements of A are an edge feature, representing the level of influence/dependency between two vertices. In our case, these elements represent the "closeness" of nodes, a concept that is natural in a geometrical multigrid problem but needs to be defined analogously in an algebraic multigrid problem. The GN then needs to output a graph with updated edge features. The output graph has the same dimensions as the input graph. But with a few simple steps, the new edge features can be used to form the non-zero elements of P.

AMG methods are originally developed for As that are M-matrices, which are matrices that are symmetric and positive definite with positive entries on the diagonal and nonpositive off-diagonal entries. This assumption is not necessary for AMG to work, but standard AMG is less likely to be effective if As are far from M-matrices. In Luz's work, As are chosen to be sparse symmetric positive definite or semi-definite matrices.

# Requirements

```
module load 2021
module load TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1

python -m venv amg_venv
source amg_venv/bin/activate

pip install -r requirements.txt
```

## Training

```
python graphnet_amg.py
```

## Test

```
python test_model.py
```
Model checkpoint is saved at 'training_dir/model_id'.

Tensorboard log files are outputted to 'tb_dir/model_id'.
