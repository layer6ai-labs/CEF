"""Implementations of Neural Spline Flows."""

import torch
from torch.nn import functional as F

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.nn import nets as nets
from nflows.transforms.base import CompositeTransform, MultiscaleCompositeTransform
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.transforms.conv import OneByOneConvolution
from nflows.transforms.lu import LULinear
from nflows.transforms.normalization import BatchNorm
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.reshape import SqueezeTransform


class SimpleNSF(Flow):
    def __init__(
        self,
        features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        base_distribution=None,
        include_linear=True,
        num_bins=11,
        tail_bound=10,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
    ):

        coupling_constructor = PiecewiseRationalQuadraticCouplingTransform
        mask = torch.ones(features)
        mask[::2] = -1
        if base_distribution is None:
            base_distribution = StandardNormal([features])

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
            coupling_transform = coupling_constructor(
                mask=mask,
                transform_net_create_fn=create_resnet,
                tails="linear",
                num_bins=num_bins,
                tail_bound=tail_bound,
            )
            layers.append(coupling_transform)
            mask *= -1

            if include_linear:
                linear_transform = CompositeTransform([
                    RandomPermutation(features=features),
                    LULinear(features, identity_init=True)])
                layers.append(linear_transform)

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=base_distribution,
        )


class MultiscaleNSF(Flow):
    def __init__(
        self,
        input_shape,
        hidden_channels,
        num_levels,
        num_layers_per_level,
        num_blocks_per_layer,
        base_distribution=None,
        include_linear=True,
        num_bins=11,
        tail_bound=1.0,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        presqueeze=False,
        linear_cache=False,
        out_shape=None,
    ):
        channels, height, width = input_shape
        dimensionality = channels * height * width
        coupling_constructor = PiecewiseRationalQuadraticCouplingTransform
        transform = MultiscaleCompositeTransform(num_levels, out_shape=out_shape)
        if base_distribution is None:
            base_distribution = StandardNormal([dimensionality])

        def create_resnet(in_channels, out_channels):
            return nets.ConvResidualNet(
                in_channels,
                out_channels,
                hidden_channels=hidden_channels,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers,
            )

        assert num_levels > 0, 'You need at least one level!'

        for level in range(num_levels):
            layers = []

            channels, height, width = input_shape
            if level > 0 or presqueeze:
                squeeze_transform = SqueezeTransform(factor=2)
                layers.append(squeeze_transform)

                channels *= 4
                height //= 2
                width //= 2
                input_shape = channels, height, width

            mask = torch.ones(channels)
            mask[::2] = -1

            for _ in range(num_layers_per_level):
                coupling_transform = coupling_constructor(
                    mask=mask,
                    transform_net_create_fn=create_resnet,
                    tails="linear",
                    num_bins=num_bins,
                    tail_bound=tail_bound
                )
                layers.append(coupling_transform)
                mask *= -1

                if include_linear:
                    linear_transform = OneByOneConvolution(channels, using_cache=linear_cache)
                    layers.append(linear_transform)

            input_shape = transform.add_transform(CompositeTransform(layers), input_shape)

        super().__init__(
            transform=transform,
            distribution=base_distribution,
        )
