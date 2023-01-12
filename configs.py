class ModelConfig:
        def __init__(self, mp_rounds=3, global_block=False, latent_size=64, mlp_layeres=4, concat_encoder=True):
            self.mp_rounds = mp_rounds
            self.global_block = global_block
            self.latent_size = latent_size
            self.mlp_layers = mlp_layeres
            self.concat_encoder = concat_encoder