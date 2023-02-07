# TODOs:
#	- types of training data
#	- 'error' and 'residual' are used somewhat interchangeably. For clarity I should make it more precise
#   - try a different baseline_P, e.g. the one used in Sandia report


# Graph net boosted prolongation operator for algebraic multigrid 
# (Graph net may be generalised to other ML techniques)
# This pseudocode trains a graph neural network to select the interpolation weights in the 
#	prolongation operator for transporting errors between grids in the algebraic multigrid (AMG) 
#	algorithm.
# AMG is an iterative algorithm for solving large and parse linear systems that arise from, for example,
#   discretisation of partial differential equations (PDEs). In each iteration, high frequency errors 
#	are eliminated by a classic relaxation scheme (commonly Gauss-Seidel), whereas low-frequency errors 
#	(i.e. smooth errors) are transferred onto a coarser grid where the smooth errors can be more 
#	efficiently removed by further relation sweeps. Multiple levels of such grids of increasing coarseness 
#	can be used for reducing the error. The interpolation operators transport errors between adjacent grid 
#	levels.
# The coefficient matrix of a coarse grid, A_coarse, is commonly generated by the standard Galerkin 
#	projection: A_coarse = R*A*P = P_transpose*A*P, where the restriction matrix R (fine->coarse grid) 
#	is just the transpose of the prolongation matrix P (coarse->fine grid). P and R are the interpolation 
#	operators.
# In particular, we use a neural network to help build the prolongation operator P that interpolates 
#	errors from a coarse grid to a fine one. This is done with the information of the coefficient matrix A 
#	of the finest grid, and the standard prolongation operator P_baseline generated by a traditional 
#	method. Although the sparsity pattern of P is enforced to be the same as P_baseline, the non-zero 
#	values (interpolation weights) in P are optimised through the neural network. Let matrix M describe 
#	the error propagation from one iteration to another by e_(i+1) = M * e_i, then the goal of an optimal 
#	P is to minimise M. In AMG, M = S_s1 * C * S_s2. S_s1 and S_s2 are the relaxation operators applied on 
#	the finest grid, before and after the cycle through the coarse grids, where s1 and s2 refer to the 
#	number of relaxation sweeps and are both typically 1 or 2. C represents the error reduction achieved 
#	through the coarse grids. C = I − P*[P_transpose*A*P]^(−1)*P_transpose*A. Given S_s1 and S_s2, which 
#	themselves typically do not have much room to be optimised, the choice of P determines the efficiency 
#	of error reduction per iteration. We use the Frobenius norm of M as the loss function of P. 
# The objective of the neural network can be summarised as follows: given A and P_baseline, train P to 
#	minimise M. A is a matrix of size mxm, where m is the dimension of the unknowns in the linear system, 
#	i.e. the number of nodes in the finest grid. P_baseline is a matrix of size mxn, where n is the number
#	of nodes in the coarser grid. n<m and typically n~m/2. P takes the same dimension and sparsity pattern
#	of P_baseline. The problem can be naturally represented by a graph neural network where A is the 
#	adjacency matrix. The elements of A are an edge feature, representing the level of 
#	influence/dependency between two vertices. In our case, these elements represent the "closeness" of
#	nodes, a concept that is natural in a geometrical multigrid problem but needs to be defined 
#	analogously in an algebraic multigrid problem. The GN then needs to output a graph with updated edge
#	features. The output graph has the same dimensions as the input graph. But with a few simple steps, 
#	the new edge features can be used to form the non-zero elements of P.
# AMG methods are originally developed for As that are M-matrices, which are matrices that are symmetric 
#	and positive definite with positive entries on the diagonal and nonpositive off-diagonal entries. This 
#	assumption is not necessary for AMG to work, but standard AMG is less likely to be effective if As are
#	far from M-matrices. In Luz's work, As are chosen to be sparse symmetric positive definite or 
#	semi-definite matrices.

import numpy as np
import pyamg
import tensorflow as tf
import graph_nets as gn
from model import EncodeProcessDecodeNonRecurrent
import configs
import math_utils
import tb_utils
import model
import data
from tqdm import tqdm
import uuid
import os
# from pyamg.classical import direct_interpolation
from scipy.sparse import csr_matrix


def main():
    train_config = getattr(configs, 'GRAPH_LAPLACIAN_TRAIN')
    eval_config = getattr(configs, "GRAPH_LAPLACIAN_EVAL")

    # updated to create a separate evaluation datas
    numAs_eval = 2
    eval_dataset = data.load_eval(numAs_eval, train_config.data_config)
    eval_A_graphs_tuple = csrs_to_graphs_tuple(
        eval_dataset.As, eval_dataset.coarse_nodes_list, eval_dataset.baseline_P_list
    )

    model = create_model(train_config.model_config)
    learning_rate = 1.e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    batch_size = train_config.train_config.batch_size

    run_name = str(uuid.uuid4())  # generating a unique file name
    tb_utils.create_results_dir(run_name)
    tb_utils.write_config_file(run_name, train_config)

    checkpoint_prefix = os.path.join(train_config.train_config.checkpoint_dir + "/" + run_name, "ckpt")
    log_dir = train_config.train_config.tensorboard_dir + "/" + run_name
    writer = tf.summary.create_file_writer(log_dir)
    writer.set_as_default()

    for run in range(train_config.train_config.num_runs):
        # updated to also include run, train_config.train_config.samples_per_run etc.
        run_dataset = data.create_dataset(
            train_config.train_config.samples_per_run, train_config.data_config, run
        )
        checkpoint = train_run(run_dataset, run, batch_size, train_config, model, optimizer,
                               optimizer.iterations.numpy(), checkpoint_prefix, eval_dataset,
                               eval_A_graphs_tuple, eval_config)
        # checkpoint.save(file_prefix=checkpoint_prefix)
    # generate training As
    # training_dataset = data.create_dataset(train_config.data_config)
    # total_norm = 0.
    return


def create_model(model_config):
    with tf.device('/gpu:0'):
        return model.EncodeProcessDecodeNonRecurrent(num_cores=model_config.mp_rounds,
                                                     edge_output_size=1,
                                                     node_output_size=1,
                                                     global_block=model_config.global_block,
                                                     latent_size=model_config.latent_size,
                                                     num_layers=model_config.mlp_layers,
                                                     concat_encoder=model_config.concat_encoder)


def train_run(run_dataset, run, batch_size, config, model, optimizer, iteration, checkpoint_prefix, eval_dataset,
              eval_A_graph_tuple, eval_config):

    num_As = len(run_dataset.As)
    #if num_As % batch_size != 0:
    #    raise RuntimeError("batch size must divide training data size")

    run_dataset = run_dataset.shuffle()
    num_batches = num_As // batch_size
    loop = tqdm(range(num_batches))

    for batch in loop:
        start_index = batch * batch_size
        end_index = start_index + batch_size
        batch_dataset = run_dataset[start_index:end_index]

        batch_A_graph_tuple = csrs_to_graphs_tuple(batch_dataset.As, batch_dataset.coarse_nodes_list,
                                                   batch_dataset.baseline_P_list)

        # converting the list of P_baseline's to Tensor format
        P_baseline_tensor_list = [tf.convert_to_tensor(P_baseline.toarray(), dtype=tf.float64)
                                  for P_baseline in batch_dataset.baseline_P_list]
        As_tensor = [tf.convert_to_tensor(A.toarray(), ) for A in batch_dataset.As]

        with tf.GradientTape() as tape:
            with tf.device('/gpu:0'):
                P_graphs_tuple = model(batch_A_graph_tuple)
            frob_loss, M = loss(As_tensor, batch_dataset.Ss, P_graphs_tuple, P_baseline_tensor_list,
                                batch_dataset.coarse_nodes)

        print(f"frob loss: {frob_loss.numpy()}")
        save_every = max(1000 // batch_size, 1)
        if batch % save_every == 0:
            checkpoint = save_model_and_optimizer(checkpoint_prefix, model, optimizer, iteration)
        variables = model.get_all_variables()
        grads = tape.gradient(frob_loss, variables)
        optimizer.apply_gradients(zip(grads, variables))

        tb_utils.record_tb(M, run, num_As, iteration, batch, batch_size, frob_loss, grads, loop, model, variables, eval_dataset,
                  eval_A_graph_tuple, eval_config)
    return checkpoint

def loss(A, S, P_square, P_baseline, coarse_nodes):
    P = math_utils.to_prolongation_matrix(P_square, P_baseline, coarse_nodes)
    S = tf.convert_to_tensor(S)
    M = math_utils.two_grid_error_matrix(A, P, S)

    norm = math_utils.frob_norm(M)
    return norm


def csrs_to_graphs_tuple(As_csr, coarse_nodes_list, P_baseline_list, node_feature_size=128):
    dtype = tf.float64
    # n_node is the number of rows/cols of A
    n_nodes = tf.convert_to_tensor([csr.shape[0] for csr in As_csr])
    # n_edge is the number of nonzero elements in A
    # eliminate_zeros() is used here in case there are zero values in the matrices.
    # If we are certain that there are no zero values, we can skip it because it is a costly operation.
    # In that case just do n_edge = tf.convert_to_tensor([csr.nnz for csr in As_csr]).
    # Also note that eliminate_zeros() doesn't work for floating zeros 0.0
    # To remove floating zeros one needs to use a small tolerance to find them and then set them to 0
    # n_edge = tf.convert_to_tensor([csr.eliminate_zeros().nnz for csr in As_csr])
    n_edges = tf.convert_to_tensor([csr.nnz for csr in As_csr])
    coos = [csr.tocoo() for csr in As_csr]
    senders_numpy = np.concatenate([coo.row for coo in coos])
    senders = tf.convert_to_tensor(senders_numpy)
    receivers_numpy = np.concatenate([coo.col for coo in coos])
    receivers = tf.convert_to_tensor(receivers_numpy)

    node_encodings_list = []
    for csr, coarse_nodes in zip(As_csr, coarse_nodes_list):
        # Note that using np.isin() on sets does not give expected results, but lists are fine,
        # see isin() documentation. It returns a boolean array of the size of the number of nodes
        coarse_indices = np.isin(range(csr.shape[0]), coarse_nodes, assume_unique=True)
        coarse_node_encodings = coarse_indices.astype(np.float64)
        # Luz used the ~ operator which is a shorthand for np.invert (i.e. np.bitwise_not) on ndarrays
        # but we use np.logical_not here, see https://stackoverflow.com/a/74997081
        fine_node_encodings = (np.logical_not(coarse_indices)).astype(np.float64)
        # convert node_encodings to array([[1,0],[0,1],...]) where [1,0] is coarse and [0,1] is fine
        node_encodings = np.stack([coarse_node_encodings, fine_node_encodings], axis=1)
        node_encodings_list.append(node_encodings)

    # numpy_nodes stores the node feature [1,0] or [0,1] for each node (row) or each graph (col)
    numpy_nodes = np.concatenate(node_encodings_list)
    nodes = tf.convert_to_tensor(numpy_nodes, dtype=dtype)

    edge_encodings_list = []
    for csr, coarse_nodes, P_baseline, n_node in zip(As_csr, coarse_nodes_list, P_baseline_list, n_nodes):
        # Here the P_baseline should be in csr format
        P_baseline_rows, P_baseline_cols = math_utils.P_square_sparsity_pattern(P_baseline, n_node, coarse_nodes)
        coo = csr.tocoo()
        # from Luz: construct numpy structured arrays, where each element is a tuple (row, col),
        # so that we can later use the numpy function isin()
        # dtype='i,i' specifies the format of a structured data type. The type 'i' is signed integer
        P_baseline_indices = np.core.records.fromarrays([P_baseline_rows, P_baseline_cols], dtype='i,i')
        coo_indices = np.core.records.fromarrays([coo.row, coo.col], dtype='i,i')
        # For a coo_indices record to be in the P_baseline_indices set, both row and col need to match
        same_indices = np.isin(coo_indices, P_baseline_indices, assume_unique=True)
        baseline_edges = same_indices.astype(np.float64)
        # Luz used the ~ operator which is a shorthand for np.invert (i.e. np.bitwise_not) on ndarrays
        # but we use np.logical_not here, see https://stackoverflow.com/a/74997081
        non_baseline_edges = (np.logical_not(same_indices)).astype(np.float64)

        # convert edge_encodings to array([[Aij,1,0],[Aij,0,1],...]) where [Aij,1,0] is if the edge is in P_baseline
        # and [Aij,0,1] is where the edge is not in P_baseline
        edge_encodings = np.stack([coo.data, baseline_edges, non_baseline_edges], axis=1)
        edge_encodings_list.append(edge_encodings)

    # numpy_edges stores the edge feature [Aij,1,0] or [Aij,0,1] for each edge (row) and graph (col)
    numpy_edges = np.concatenate(edge_encodings_list)
    edges = tf.convert_to_tensor(numpy_edges, dtype=dtype)

    graphs_tuple = gn.graphs.GraphsTuple(
        nodes=nodes,
        edges=edges,
        globals=None,
        receivers=receivers,
        senders=senders,
        n_node=n_nodes,
        n_edge=n_edges
    )

    graphs_tuple = gn.utils_tf.set_zero_global_features(graphs_tuple, node_feature_size, dtype=dtype)

    return graphs_tuple


def record_tb(M, run, num_As, iteration, batch, batch_size, frob_loss, grads, loop, model, variables, eval_dataset,
              eval_A_graphs_tuple, eval_config):
    batch = run * num_As + batch
    # enable eager execution
    tf.compat.v1.enable_v2_behavior()

    # A default TF 2.x summary writer
    writer = tf.summary.create_file_writer("./logs")

    record_loss_every = max(1 // batch_size, 1)
    if batch % record_loss_every == 0:
        tb_utils.record_tb_loss(writer, iteration, frob_loss)

    # record_params_every = max(300 // batch_size, 1)
    # if batch % record_params_every == 0:
    #     record_tb_params(writer, batch_size, grads, loop, variables)
    #
    # record_spectral_every = max(300 // batch_size, 1)
    # if batch % record_spectral_every == 0:
    #     record_tb_spectral_radius(writer, M, model, eval_dataset, eval_A_graphs_tuple, eval_config)


def save_model_and_optimizer(checkpoint_prefix, model, optimizer, iteration):
    variables = model.get_all_variables()
    variables_dict = {variables.name: variable for variable in variables}
    checkpoint = None
    return checkpoint

if __name__ == '__main__':
    main()
