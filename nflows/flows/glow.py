"""Implementations of Glow."""

import torch
from torch.nn import functional as F

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.nn import nets as nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import (
    AdditiveCouplingTransform,
    AffineCouplingTransform,
)
from nflows.transforms.lu import LULinear
from nflows.transforms.normalization import ActNorm
from nflows.transforms.permutations import RandomPermutation


class SimpleGlow(Flow):
    """An simplified version of Glow for 1-dim inputs.

    This implementation uses 1-dim checkerboard masking but doesn't use multi-scaling.

    Reference:
    > D. Kingma et. al., Glow: Generative flow with invertible 1x1 convolutions, NeurIPS 2018.
    """

    def __init__(
        self,
        features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        use_volume_preserving=False,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
    ):

        if use_volume_preserving:
            coupling_constructor = AdditiveCouplingTransform
        else:
            coupling_constructor = AffineCouplingTransform

        mask = torch.ones(features)
        mask[::2] = -1

        def create_resnet(in_features, out_features):
            return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_features,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers,
            )

        layers = []
        for _ in range(num_layers):
            actnorm = ActNorm(features)
            layers.append(actnorm)

            linear_transform = CompositeTransform([
                RandomPermutation(features=features),
                LULinear(features, identity_init=True)])
            layers.append(linear_transform)

            transform = coupling_constructor(
                mask=mask, transform_net_create_fn=create_resnet
            )
            layers.append(transform)
            mask *= -1


        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([features]),
        )
