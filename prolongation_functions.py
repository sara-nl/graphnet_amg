import tensorflow as tf
from scipy.sparse import csr_matrix, coo_matrix

from math_utils import to_prolongation_matrix_tensor
from graphnet_amg import csrs_to_graphs_tuple, graphs_tuple_to_sparse_tensor



def model(A, coarse_nodes, baseline_P, C, graph_model):
    with tf.device(":/gpu:0"):
        graphs_tuple = csrs_to_graphs_tuple([A], coarse_nodes_list=[coarse_nodes],
                                            P_baseline_list=[baseline_P], 
                                            node_feature_size=128)
        output_graph = graph_model(graphs_tuple)

    P_square_tensor = graphs_tuple_to_sparse_tensor(output_graph)
    nodes_tensor = tf.squeeze(output_graph.nodes)
    nodes = nodes_tensor.numpy()

    P_square_csr = sparse_tensor_to_csr(P_square_tensor)
    P_tensor = to_prolongation_matrix_tensor(P_square_csr, coarse_nodes, baseline_P, nodes, normalize_rows=True, normalize_rows_by_node=False)

    return sparse_tensor_to_csr(P_tensor)


def baseline(A, splitting, baseline_P, C):
    return baseline_P