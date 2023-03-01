import os
from functools import partial
import fire

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from scipy.sparse import csr_matrix, coo_matrix

import configs
import tb_utils
import data
from graphnet_amg import create_model, csrs_to_graphs_tuple

from prolongation_functions import model, baseline
import pyamg
from ruge_stuben_custom_solver import ruge_stuben_custom_solver


def test_model(model_name='fragrant-dawn-19_3e6bcd7d-41cd-4dbe-aa13-760a06db0d3c', seed=1):
    if model_name is None:
        raise RuntimeError("model name required")
    #model_name = str(model_name)

    # fix random seeds for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)

    test_configs = getattr(configs, 'GRAPH_LAPLACIAN_TEST')  

    train_config = test_configs.train_config
    data_config = test_configs.data_config
    model_config = test_configs.model_config
    test_config = test_configs.test_config

    model = get_model(model_name, model_config, train_config)

    for size in test_config.test_sizes:
        test_size(model_name, model, size, test_config, train_config, data_config)


# function for testing
def test_size(model_name, graph_model, size, test_config, train_config, data_config):
    '''
    Function for testing the trained model 
    '''
    model_prolongation = partial(model, graph_model=graph_model)
    baseline_prolongation = baseline

    model_errors_div_diff = []
    baseline_errors_div_diff = []

    fp_threshold = test_config.fp_threshold
    strength = test_config.strength
    presmoother = test_config.presmoother
    postsmoother = test_config.postsmoother
    coarse_solver = test_config.coarse_solver

    cycle = test_config.cycle
    splitting = test_config.splitting
    dist = test_config.dist
    num_runs = test_config.num_runs
    max_levels = test_config.max_levels
    iterations = test_config.iterations
    load_data = test_config.load_data

    block_periodic = False
    root_num_blocks = 1

    results_dir = train_config.results_dir

    
    if dist == 'sparse_block_circulant':
        As = get_test_data(size, test_config)
    elif dist == 'lognormal_complex_fem':
        As = np.load(f"test_data_dir/fe_hole_logn_num_As_{100}_num_points_{size}.npy") #TODO
    else:
        raise NotImplementedError()

    for i in tqdm(range(num_runs)):
        A = As[i]

        num_unknowns = A.shape[0]
        x0 = np.random.normal(loc=0.0, scale=1.0, size=num_unknowns)
        b = np.zeros((A.shape[0]))

        model_residuals = []
        baseline_residuals = []

        model_solver = ruge_stuben_custom_solver(A, model_prolongation,         #TODO to check when model checkpoint is available
                                                 strength=strength,
                                                 presmoother=presmoother,
                                                 postsmoother=postsmoother,
                                                 keep=True, max_levels=max_levels,
                                                 CF=splitting,
                                                 coarse_solver=coarse_solver)

        _ = model_solver.solve(b, x0=x0, tol=0.0, maxiter=iterations, cycle=cycle,
                               residuals=model_residuals)
        model_residuals = np.array(model_residuals)
        model_residuals = model_residuals[model_residuals > fp_threshold]
        model_factor = model_residuals[-1] / model_residuals[-2]
        model_errors_div_diff.append(model_factor)

        baseline_solver = ruge_stuben_custom_solver(A, baseline_prolongation,
                                                    strength=strength,
                                                    presmoother=presmoother,
                                                    postsmoother=postsmoother,
                                                    keep=True, max_levels=max_levels,
                                                    CF=splitting,
                                                    coarse_solver=coarse_solver)
        baseline_solver = None
        _ = baseline_solver.solve(b, x0=x0, tol=0.0, maxiter=iterations, cycle=cycle,
                                  residuals=baseline_residuals)
        baseline_residuals = np.array(baseline_residuals)
        baseline_residuals = baseline_residuals[baseline_residuals > fp_threshold]
        baseline_factor = baseline_residuals[-1] / baseline_residuals[-2]
        baseline_errors_div_diff.append(baseline_factor)

    model_errors_div_diff = np.array(model_errors_div_diff)
    baseline_errors_div_diff = np.array(baseline_errors_div_diff)
    model_errors_div_diff_mean = np.mean(model_errors_div_diff)
    model_errors_div_diff_std = np.std(model_errors_div_diff)
    baseline_errors_div_diff_mean = np.mean(baseline_errors_div_diff)
    baseline_errors_div_diff_std = np.std(baseline_errors_div_diff)

    if type(splitting) == tuple:
        splitting_str = splitting[0] + '_'+ '_'.join([f'{key}_{value}' for key, value in splitting[1].items()])
    else:
        splitting_str = splitting
    results_file = open(
        f"results/{model_name}/{dist}_{num_unknowns}_cycle_{cycle}_max_levels_{max_levels}_split_{splitting_str}_results.txt",
        'w')
    print(f"cycle: {cycle}, max levels: {max_levels}", file=results_file)
    print(f"asymptotic error factor model: {model_errors_div_diff_mean:.4f} ± {model_errors_div_diff_std:.5f}",
          file=results_file)

    print(f"asymptotic error factor baseline: {baseline_errors_div_diff_mean:.4f} ± {baseline_errors_div_diff_std:.5f}",
          file=results_file)
    model_success_rate = sum(model_errors_div_diff < baseline_errors_div_diff) / num_runs
    print(f"model success rate: {model_success_rate}",
          file=results_file)

    print(f"num unknowns: {num_unknowns}")
    print(f"asymptotic error factor model: {model_errors_div_diff_mean:.4f} ± {model_errors_div_diff_std:.5f}")
    print(f"asymptotic error factor baseline: {baseline_errors_div_diff_mean:.4f} ± {baseline_errors_div_diff_std:.5f}")
    print(f"model success rate: {model_success_rate}")

    results_file.close()



## utils for loading the model 
def get_model(model_name, model_config, train_config, train=False):
    dummy_input = data.As_poisson_grid(1, 7 ** 2)[0]
    checkpoint_dir = train_config.checkpoint_dir + model_name
    graph_model, optimizer, global_step = load_model(checkpoint_dir, dummy_input, 
                                                     model_config, train_config,
                                                     get_optimizer=train)
    if train:
        return graph_model, optimizer, global_step
    else:
        return graph_model


def load_model(checkpoint_dir, dummy_input, model_config, train_config, get_optimizer=True):
    model = create_model(model_config)

    # we have to use the model at least once to get the list of variables As_csr, coarse_nodes_list, P_baseline_list, node_feature_size=128)
    model(csrs_to_graphs_tuple([dummy_input], coarse_nodes_list=np.array([[0, 1]]), 
                               P_baseline_list=[ csr_matrix(dummy_input.toarray()[:, [0, 1]]) ] ))   

    variables = model.variables
    variables_dict = {variable.name: variable for variable in variables}
    if get_optimizer:
        #decay_steps = 100
        #decay_rate = 1.0
        #learning_rate = tf.train.exponential_decay(train_config.learning_rate, global_step, decay_steps, decay_rate)
        optimizer = tf.keras.optimizers.Adam(learning_rate=train_config.learning_rate)
        global_step = optimizer.iterations.numpy()

        checkpoint = tf.train.Checkpoint(**variables_dict, optimizer=optimizer)    
    else:
        optimizer = None
        global_step = None
        checkpoint = tf.train.Checkpoint(**variables_dict)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint is None:
        raise RuntimeError(f'results dir {checkpoint_dir} does not exist')
    checkpoint.restore(latest_checkpoint)
    return model, optimizer, global_step


def get_test_data(size, test_config):
    filename = f"{test_config.data_dir}test_num_As_{test_config.num_As}_num_points_{size}.npy"
    if os.path.exists(filename):
        As = data.load_from_file(filename)
    else:
        test_config.num_unknowns = size
        As = data.generate_in_one_file(test_config.num_As, test_config, filename)

    return As


if __name__ == '__main__':
    tb_utils.config_tf()

    #fire.Fire(test_model)  # python test_model.py -model-name 12345  
    test_model()
    print('test done')