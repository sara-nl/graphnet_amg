class ModelConfig:
    def __init__(
        self,
        mp_rounds=3,
        global_block=False,
        latent_size=64,
        mlp_layeres=4,
        concat_encoder=True,
    ):
        self.mp_rounds = mp_rounds
        self.global_block = global_block
        self.latent_size = latent_size
        self.mlp_layers = mlp_layeres
        self.concat_encoder = concat_encoder


class DataConfig:
    def __init__(
        self,
        dist="sparse_block_circulant",
        num_As=1000,
        num_unknowns=8**2,
        num_blocks=4,
        splitting="CLJP",
        load_data="True",
        save_data=True,
        data_dir=f"/home/monicar/DL4NS/learning-amg/my_amg/data_dir/",
    ):
        self.dist = dist
        self.num_As = num_As
        self.num_unknowns = num_unknowns
        self.num_blocks = num_blocks
        self.splitting = splitting
        self.load_data = load_data
        self.save_data = save_data
        self.data_dir = data_dir


class TrainConfig:
    def __init__(
        self,
        samples_per_run=256,
        num_runs=1,
        batch_size=32,
        learning_rate=3e-3,
        coarsen=False,
        checkpoint_dir=f"/home/monicar/DL4NS/learning-amg/my_amg/training_dir/",
        tensorboard_dir="./tb_dir",
        results_dir= f"/home/monicar/DL4NS/learning-amg/my_amg/results/",
        load_model=False,
    ):
        self.samples_per_run = samples_per_run
        self.num_runs = num_runs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.coarsen = coarsen
        self.checkpoint_dir = checkpoint_dir
        self.tensorboard_dir = tensorboard_dir
        self.results_dir = results_dir
        self.load_model = load_model


class TestConfig:
    def __init__(self, 
                 dist='sparse_block_circulant', 
                 splitting='CLJP',
                 test_sizes=(1024, 2048),    #n = 1024, where n = b^2 * c -> for testing b = 1, c = 1024
                 load_data=True, 
                 save_data = True,
                 num_runs=100, 
                 cycle='W',
                 max_levels=12, 
                 iterations=81, 
                 fp_threshold=1e-10, 
                 strength=('classical', {'theta': 0.25}),
                 presmoother=('gauss_seidel', {'sweep': 'forward', 'iterations': 1}),
                 postsmoother=('gauss_seidel', {'sweep': 'forward', 'iterations': 1}),
                 coarse_solver='pinv2',
                 data_dir=f"/home/monicar/DL4NS/learning-amg/my_amg/data_dir/",
                 num_As=100,
                 num_unknowns=1024,
                 num_blocks=1,
                 ):
        self.dist = dist
        self.splitting = splitting
        self.test_sizes = test_sizes
        self.load_data = load_data
        self.save_data = save_data
        self.num_runs = num_runs
        self.cycle = cycle
        self.max_levels = max_levels
        self.iterations = iterations
        self.fp_threshold = fp_threshold
        self.strength = strength
        self.presmoother = presmoother
        self.postsmoother = postsmoother
        self.coarse_solver = coarse_solver
        self.data_dir = data_dir
        self.num_As = num_As
        self.num_unknowns = num_unknowns
        self.num_blocks = num_blocks


class Config:
    def __init__(self):
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        self.train_config = TrainConfig()
        self.test_config = TestConfig()


GRAPH_LAPLACIAN_TRAIN = Config()
GRAPH_LAPLACIAN_EVAL = Config()
GRAPH_LAPLACIAN_EVAL.data_config.num_unknowns = 128
GRAPH_LAPLACIAN_TEST = Config()
