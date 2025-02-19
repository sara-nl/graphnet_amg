import sonnet as snt
import graph_nets as gn
import tensorflow as tf
from graph_nets import modules
from functools import partial


class EncodeProcessDecodeNonRecurrent(snt.Module):
    """
    similar to EncodeProcessDecode, but with non-recurrent core
    see docs for EncodeProcessDecode
    """

    def __init__(self,
                 num_cores=3,
                 edge_output_size=None,
                 node_output_size=None,
                 global_output_size=None,
                 global_block=True,
                 latent_size=16,
                 num_layers=2,
                 concat_encoder=True,
                 name="EncodeProcessDecodeNonRecurrent"):
        super(EncodeProcessDecodeNonRecurrent, self).__init__(name=name)
        self._num_cores = num_cores
        self._edge_output_size = edge_output_size
        self._node_output_size = node_output_size
        self._global_output_size = global_output_size
        self._global_block = global_block
        self._latent_size = latent_size
        self._num_layers = num_layers
        self._concat_encoder = concat_encoder

        self._encoder = MLPGraphIndependent(self._latent_size, self._num_layers)
        self._cores = [MLPGraphNetwork(latent_size=self._latent_size, num_layers=self._num_layers,
                                       global_block=self._global_block) for _ in range(self._num_cores)]
        self._decoder = MLPGraphIndependent(self._latent_size, self._num_layers)
        self.concat_encoder = self._concat_encoder
        # Transforms the outputs into the appropriate shapes.
        if self._edge_output_size is None:
            edge_fn = None
        else:
            edge_fn = lambda: snt.Linear(self._edge_output_size, name="edge_output")
        if self._node_output_size is None:
            node_fn = None
        else:
            node_fn = lambda: snt.Linear(self._node_output_size, name="node_output")
        if self._global_output_size is None:
            global_fn = None
        else:
            global_fn = lambda: snt.Linear(self._global_output_size, name="global_output")

        self._output_transform = modules.GraphIndependent(edge_fn, node_fn, global_fn)


    def __call__(self, input_op):
        latent = self._encoder(input_op)
        latent0 = latent
        for i in range(len(self._cores)):
            if self.concat_encoder:
                core_input = gn.utils_tf.concat([latent0, latent], axis=1)
            else:
                core_input = latent
            latent = self._cores[i](core_input)
        return self._output_transform(self._decoder(latent))


class MLPGraphNetwork(snt.Module):
    """GraphNetwork with MLP edge, node, and global models."""

    def __init__(self,
                 latent_size=16,
                 num_layers=2,
                 global_block=True,
                 last_round=False,
                 name="MLPGraphNetwork"):
        super(MLPGraphNetwork, self).__init__(name=name)
        self._latent_size = latent_size
        self._num_layers = num_layers
        self._global_block = global_block
        self._last_round = last_round

        partial_make_mlp_model = partial(make_mlp_model, self._latent_size, self._num_layers,
                                         last_round_edges=False)
        if self._last_round:
            partial_make_mlp_model_edges = partial(make_mlp_model, self._latent_size, self._num_layers,
                                                   last_round_edges=True)
        else:
            partial_make_mlp_model_edges = partial_make_mlp_model
        if self._global_block:
            self._network = modules.GraphNetwork(partial_make_mlp_model_edges, partial_make_mlp_model,
                                                 partial_make_mlp_model,
                                                 edge_block_opt={
                                                     "use_globals": True
                                                 },
                                                 node_block_opt={
                                                     "use_globals": True
                                                 },
                                                 global_block_opt={
                                                     "use_globals": True,
                                                     "edges_reducer": tf.math.unsorted_segment_mean,
                                                     "nodes_reducer": tf.math.unsorted_segment_mean
                                                 })
        else:
            self._network = modules.GraphNetwork(partial_make_mlp_model_edges, partial_make_mlp_model,
                                                 make_identity_model,
                                                 edge_block_opt={
                                                     "use_globals": False
                                                 },
                                                 node_block_opt={
                                                     "use_globals": False
                                                 },
                                                 global_block_opt={
                                                     "use_globals": False,
                                                 })
    def __call__(self, inputs):
        return self._network(inputs)


class MLPGraphIndependent(snt.Module):
    """GraphIndependent with MLP edge, node, and global models."""

    def __init__(self,
                 latent_size=16,
                 num_layeres=2,
                 name="MLPGraphIndependent"):
        super(MLPGraphIndependent, self).__init__(name=name)
        self._latent_size = latent_size
        self._num_layers = num_layeres

        partial_make_mlp_model = partial(make_mlp_model, self._latent_size, self._num_layers,
                                         last_round_edges=False)
        self._network = modules.GraphIndependent(
            edge_model_fn=partial_make_mlp_model,
            node_model_fn=partial_make_mlp_model,
            global_model_fn=partial_make_mlp_model)


    def __call__(self, inputs):
        return self._network(inputs)


def make_mlp_model(latent_size=16, num_layers=2, last_round_edges=False):
    """Instantiates a new MLP, followed by LayerNorm.

  The parameters of each new MLP are not shared with others generated by
  this function.

  Returns:
    A Sonnet module which contains the MLP and LayerNorm.
  """
    if last_round_edges:
        return snt.nets.MLP([latent_size] * num_layers + [1], activate_final=False)
    else:
        return snt.Sequential([
            snt.nets.MLP([latent_size] * num_layers, activate_final=False)
        ])


class IdentityModule(snt.Module):
    def __call__(self, inputs):
        return tf.identity(inputs)


def make_identity_model():
    return IdentityModule()