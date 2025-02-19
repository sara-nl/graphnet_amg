# TODOs:
#   - try a different baseline_P, e.g. the one used in Sandia report


# Graph net boosted prolongation operator for algebraic multigrid 
# This program trains a graph neural network to select the interpolation weights in the 
#	prolongation operator for transporting errors between grids in the algebraic multigrid (AMG) 
#	algorithm.

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
import wandb


def main():
    train_config = getattr(configs, 'GRAPH_LAPLACIAN_TRAIN')
    eval_config = getattr(configs, "GRAPH_LAPLACIAN_EVAL")
    wandb.init(project=train_config.train_config.wandb_project,
               entity=train_config.train_config.wandb_user, sync_tensorboard=True)

    # create a separate evaluation dataset
    numAs_eval = 1
    eval_dataset = data.create_dataset(numAs_eval, eval_config.data_config, eval=True)
    eval_A_graphs_tuple = csrs_to_graphs_tuple(
        eval_dataset.As, eval_dataset.coarse_nodes_list, eval_dataset.baseline_P_list
    )

    model = create_model(train_config.model_config)
    learning_rate = train_config.train_config.learning_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    batch_size = train_config.train_config.batch_size

    run_name = wandb.run.name + '_' + str(uuid.uuid4())  # generating a unique file name
    tb_utils.create_results_dir(run_name)
    tb_utils.write_config_file(run_name, train_config)

    checkpoint_prefix = os.path.join(train_config.train_config.checkpoint_dir + "/" + run_name, "ckpt")
    log_dir = train_config.train_config.tensorboard_dir + "/" + run_name
    writer = tf.summary.create_file_writer(log_dir)
    # writer.set_as_default()

    for run in range(train_config.train_config.num_runs):
        # create dataset for the run
        run_dataset = data.create_dataset(
            train_config.train_config.samples_per_run, train_config.data_config, run=run
        )
        checkpoint = train_run(run_dataset, run, batch_size, train_config, model, optimizer,
                               optimizer.iterations.numpy(), checkpoint_prefix, eval_dataset,
                               eval_A_graphs_tuple, eval_config, writer)
        checkpoint.save(file_prefix=checkpoint_prefix)
    return


def create_model(model_config):
    with tf.device('/GPU:0'):
        return model.EncodeProcessDecodeNonRecurrent(num_cores=model_config.mp_rounds,
                                                     edge_output_size=1,
                                                     node_output_size=1,
                                                     global_block=model_config.global_block,
                                                     latent_size=model_config.latent_size,
                                                     num_layers=model_config.mlp_layers,
                                                     concat_encoder=model_config.concat_encoder)


def train_run(run_dataset, run, batch_size, config, model, optimizer, iteration, checkpoint_prefix, eval_dataset,
              eval_A_graph_tuple, eval_config, writer):

    num_As = len(run_dataset.As)
    if num_As % batch_size != 0:
        raise RuntimeError("batch size must divide training data size")

    run_dataset = run_dataset.shuffle()
    num_batches = num_As // batch_size
    loop = tqdm(range(num_batches))

    for batch in loop:
        start_index = batch * batch_size
        end_index = start_index + batch_size
        batch_dataset = run_dataset[start_index:end_index]

        batch_A_graph_tuple = csrs_to_graphs_tuple(batch_dataset.As, batch_dataset.coarse_nodes_list,
                                                   batch_dataset.baseline_P_list)

        with tf.GradientTape() as tape:
            with tf.device('/GPU:0'):
                batch_P_graphs_tuple = model(batch_A_graph_tuple)
            frob_loss, frob_baseline, M, M_baseline = loss(batch_dataset, batch_A_graph_tuple, batch_P_graphs_tuple)

        print(f"frob loss: {frob_loss.numpy()}")
        print(f"frob baseline: {frob_baseline.numpy()}")
        save_every = max(1000 // batch_size, 1)
        if batch % save_every == 0:
            checkpoint = save_model_and_optimizer(checkpoint_prefix, model, optimizer, iteration)
        variables = model.variables
        grads = tape.gradient(frob_loss, variables)
        optimizer.apply_gradients(zip(grads, variables))

        # tb_utils.record_tb(M, run, num_As, iteration, batch, batch_size, frob_loss, grads, loop, model, variables, eval_dataset,
        #           eval_A_graph_tuple, eval_config)

        with writer.as_default():
            tb_utils.record_tb(M, run, num_As, iteration, batch, batch_size, frob_loss, grads, loop, model, variables, eval_dataset, eval_A_graph_tuple, eval_config)
        writer.flush()

        # validation
        eval_loss, eval_M = validation(model, eval_dataset, eval_A_graph_tuple, eval_config)
        with writer.as_default():
            tb_utils.record_tb_eval(M, run, num_As, iteration, batch, batch_size, eval_loss, eval_M)
        writer.flush()
    return checkpoint

def validation(model, eval_dataset, eval_A_graphs_tuple, eval_config):
    with tf.device('/GPU:0'):
        eval_P_graphs_tuple = model(eval_A_graphs_tuple)
    eval_loss, eval_frob_baseline, eval_M, eval_M_baseline = loss(eval_dataset, eval_A_graphs_tuple, eval_P_graphs_tuple)

    return eval_loss, eval_M

def loss(dataset, A_graphs_tuple, P_graphs_tuple):
    As = dataset.As
    Ps_baseline = dataset.baseline_P_list
    Ps_square, nodes_list = graphs_tuple_to_sparse_matrices(P_graphs_tuple, True)

    # converting the list of P_baseline's to Tensor format
    # P_baseline_tensor_list = [tf.convert_to_tensor(P_baseline.toarray(), dtype=tf.float64)
    #                           for P_baseline in batch_dataset.baseline_P_list]
    # As_tensor = [tf.convert_to_tensor(A.toarray(), ) for A in batch_dataset.As]

    batch_size = len(dataset.coarse_nodes_list)
    total_norm = tf.Variable(0.0, dtype=tf.float64)
    total_norm_baseline = tf.Variable(0.0, dtype=tf.float64)
    for i in range(batch_size):
        #A = tf.sparse.to_dense(As[i])
        A_tensor = tf.convert_to_tensor(As[i].toarray(), dtype=tf.float64)
        P_square = Ps_square[i]
        coarse_nodes = dataset.coarse_nodes_list[i]
        P_baseline = tf.convert_to_tensor(dataset.baseline_P_list[i].toarray(), dtype=tf.float64)
        nodes = nodes_list[i]
        P = math_utils.to_prolongation_matrix_tensor(P_square, coarse_nodes, P_baseline, nodes)
        S = tf.convert_to_tensor(dataset.Ss[i], dtype=tf.float64)
        M = math_utils.two_grid_error_matrix(A_tensor, P, S)
        M_baseline = math_utils.two_grid_error_matrix(A_tensor, P_baseline, S)
        norm = math_utils.frob_norm(M)
        norm_baseline = math_utils.frob_norm(M_baseline)
        total_norm = total_norm + norm
        total_norm_baseline = total_norm_baseline + norm_baseline
    return total_norm / batch_size, total_norm_baseline / batch_size, M, M_baseline


def graphs_tuple_to_sparse_matrices(graphs_tuple, return_nodes=False):
    num_graphs = int(graphs_tuple.n_node.shape[0])
    graphs = [gn.utils_tf.get_graph(graphs_tuple, i)
              for i in range(num_graphs)]

    matrices = [graphs_tuple_to_sparse_tensor(graph) for graph in graphs]

    if return_nodes:
        nodes_list = [tf.squeeze(graph.nodes) for graph in graphs]
        return matrices, nodes_list
    else:
        return matrices


def graphs_tuple_to_sparse_tensor(graphs_tuple):
    senders = graphs_tuple.senders
    receivers = graphs_tuple.receivers
    indices = tf.cast(tf.stack([senders, receivers], axis=1), tf.int64)

    # first element in the edge feature is the value, the other elements are metadata
    values = tf.squeeze(graphs_tuple.edges[:, 0])

    shape = tf.concat([graphs_tuple.n_node, graphs_tuple.n_node], axis=0)
    shape = tf.cast(shape, tf.int64)

    matrix = tf.sparse.SparseTensor(indices, values, shape)
    # reordering is required because the pyAMG coarsening step does not preserve indices order
    matrix = tf.sparse.reorder(matrix)

    return matrix


def csrs_to_graphs_tuple(As_csr, coarse_nodes_list, P_baseline_list, node_feature_size=128):
    dtype = tf.float64
    # n_node is the number of rows/cols of A
    n_nodes = tf.convert_to_tensor([csr.shape[0] for csr in As_csr])
    # n_edge is the number of nonzero elements in A
    # eliminate_zeros() can be used here in case there are zero values in the matrices:
    # n_edge = tf.convert_to_tensor([csr.eliminate_zeros().nnz for csr in As_csr])
    # If we are certain that there are no zero values, we can skip it because it is a costly operation.
    # In that case just do n_edge = tf.convert_to_tensor([csr.nnz for csr in As_csr])
    # Also note that eliminate_zeros() doesn't work for floating zeros 0.0
    # To remove floating zeros one needs to use a small tolerance to find them and then set them to 0
    # eliminate_zeros() is not used for now
    n_edges = tf.convert_to_tensor([csr.nnz for csr in As_csr])
    coos = [csr.tocoo() for csr in As_csr]
    senders_numpy = np.concatenate([coo.row for coo in coos])
    senders = tf.convert_to_tensor(senders_numpy)
    receivers_numpy = np.concatenate([coo.col for coo in coos])
    receivers = tf.convert_to_tensor(receivers_numpy)

    # see the source of _concatenate_data_dicts for details, but basically when we encode multiple graphs
    #   into a GraphsTuple object, the dictionary expects receivers and senders to be NOT the indices of receivers
    #   and senders of its own graph, but the indices plus the edge index. Basically this makes the receivers and
    #   senders to be unique values in the GraphsTuple
    offsets = gn.utils_tf._compute_stacked_offsets(n_nodes, n_edges)
    senders += offsets
    receivers += offsets

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

    # numpy_nodes stores the node feature [1,0] or [0,1] for each node in all graphs
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
        # edge_encodings_list has dimensions num_As x num_edges x 3
        # where num_edges is number of non-zero elements in A, and 3 is [Aij, 1, 0] or [Aij, 0, 1]
        # [Aij, 1, 0] for when the edge is in the baseline P, otherwise not
        edge_encodings_list.append(edge_encodings)

    # numpy_edges stores the edge feature [Aij,1,0] or [Aij,0,1] for each edge in all graphs
    numpy_edges = np.concatenate(edge_encodings_list)
    edges = tf.convert_to_tensor(numpy_edges, dtype=dtype)

    # RECEIVERS: the index is absolute (in other words, cumulative), i.e.
    #     `graphs.RECEIVERS` take value in `[0, n_nodes]`. For instance, an edge
    #     connecting the vertices with relative indices 2 and 3 in the second graph of
    #     the batch would have a `RECEIVERS` value of `3 + graph.N_NODE[0]`.
    # Likewise for SENDERS
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


def save_model_and_optimizer(checkpoint_prefix, model, optimizer, iteration):
    variables = model.variables
    variables_dict = {variable.name: variable for variable in variables}
    checkpoint = tf.train.Checkpoint(**variables_dict, optimizer=optimizer)
    if not os.path.exists(checkpoint_prefix):
        os.makedirs(checkpoint_prefix)
    checkpoint.save(file_prefix=checkpoint_prefix)
    return checkpoint

if __name__ == '__main__':
    tb_utils.config_tf()
    tb_utils.get_available_devices()
    main()
