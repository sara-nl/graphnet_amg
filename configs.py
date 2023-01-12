class ModelConfig:
    def __init__(self, mp_rounds=3, global_block=False, latent_size=64, mlp_layeres=4, concat_encoder=True):
        self.mp_rounds = mp_rounds
        self.global_block = global_block
        self.latent_size = latent_size
        self.mlp_layers = mlp_layeres
        self.concat_encoder = concat_encoder


class DataConfig:
    def __init__(self, dist='sparse_block_circulant',
                 num_unknowns=8 ** 2, num_blocks=4, splitting='CLJP',
                 load_data='True', save_data=True):
        self.dist = dist
        self.num_unknowns = num_unknowns
        self.num_blocks = num_blocks
        self.splitting = splitting
        self.load_data = load_data
        self.save_data = save_data


class Config:
    def __init__(self):
        self.data_config = DataConfig()


GRAPH_LAPLACIAN_TRAIN = Config()