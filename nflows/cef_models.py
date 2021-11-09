from nflows import transforms, distributions, flows


class SphereCEFlow(flows.InjectiveFlow):
    def __init__(self):
        n = 3
        m = 2

        conf_embedding = transforms.CompositeTransform([
            transforms.ConformalScaleShift(n, m),
            transforms.Orthogonal(n),
            transforms.SpecialConformal(n, m),
            transforms.Pad(n, m),
        ])
        base_flow = flows.SimpleNSF(
            features=m,
            hidden_features=16,
            num_layers=2,
            num_blocks_per_layer=1)

        super().__init__(embedding=conf_embedding, distribution=base_flow, conformal=True)

class MixturePlaneCEFlow(flows.InjectiveFlow):
    def __init__(self, base_flow_class=flows.SimpleNSF):
        n = 3
        m = 2

        conf_embedding = transforms.CompositeTransform([
            transforms.Orthogonal1x1Conv(n, m),
            transforms.Flatten((m, 1, 1))
        ])
        base_flow = base_flow_class(
            features=m, hidden_features=128, num_layers=3, num_blocks_per_layer=2)

        super().__init__(embedding=conf_embedding, distribution=base_flow, conformal=True)


class MNISTMFlow(flows.InjectiveFlow):
    def __init__(self, m):
        embedding = transforms.CompositeTransform([
            flows.MultiscaleNSF(
                input_shape=(1, 32, 32),
                hidden_channels=64,
                num_levels=3,
                num_layers_per_level=3,
                num_blocks_per_layer=3,
                presqueeze=True,
            )._transform,
            transforms.LULinear(1024),
            transforms.Pad(1024, m),
        ])
        base_flow = (
            flows.SimpleNSF(features=m, hidden_features=512, num_layers=8, num_blocks_per_layer=3))

        super().__init__(embedding=embedding, distribution=base_flow, conformal=False)


class MNISTCEFlow(flows.InjectiveFlow):
    def __init__(self, m):
        channels = 1

        conf_embedding = transforms.CompositeTransform([
            transforms.ConformalScaleShift(channels, m),
            transforms.HouseholderConv(channels, kernel_size=8),

            transforms.ConformalScaleShift(channels*64, m),
            transforms.ConditionalOrthogonal1x1Conv(channels*64),

            transforms.ConformalScaleShift(channels*64, m),
            transforms.SqueezeTransform(factor=4),

            transforms.Orthogonal1x1Conv(channels*1024, m),

            transforms.Flatten((m, 1, 1))
        ])
        base_flow = (
            flows.SimpleNSF(features=m, hidden_features=512, num_layers=8, num_blocks_per_layer=3))

        super().__init__(embedding=conf_embedding, distribution=base_flow, conformal=True)


class Cifar10MFlow(flows.InjectiveFlow):
    def __init__(self, m):
        assert m % 64 == 0, 'This model requires manifold dimension to be divisible by 64'

        embedding = transforms.CompositeTransform([
            flows.MultiscaleNSF(
                input_shape=(3, 32, 32),
                out_shape=(8, 8),
                hidden_channels=64,
                num_levels=3,
                num_layers_per_level=3,
                num_blocks_per_layer=3,
                presqueeze=True,
            )._transform,
            transforms.OneByOneConvolution(48),
            transforms.Pad(48, m // 64),
            transforms.Flatten((m // 64, 8, 8)),
        ])
        base_flow = (
            flows.SimpleNSF(features=m, hidden_features=512, num_layers=8, num_blocks_per_layer=3))

        super().__init__(embedding=embedding, distribution=base_flow, conformal=False)


class Cifar10CEFlow(flows.InjectiveFlow):
    def __init__(self, m):
        channels = 3

        conf_embedding = transforms.CompositeTransform([
            transforms.ConformalScaleShift(channels, m),
            transforms.HouseholderConv(channels, kernel_size=8),

            transforms.ConformalScaleShift(channels*64, m),
            transforms.ConditionalOrthogonal1x1Conv(channels*64),

            transforms.ConformalScaleShift(channels*64, m),
            transforms.SqueezeTransform(factor=4),

            transforms.Orthogonal1x1Conv(channels*1024, m),

            transforms.Flatten((m, 1, 1))
        ])
        base_flow = (
            flows.SimpleNSF(features=m, hidden_features=512, num_layers=8, num_blocks_per_layer=3))

        super().__init__(embedding=conf_embedding, distribution=base_flow, conformal=True)


class CelebAMFlow(flows.InjectiveFlow):
    def __init__(self, deeper=False):
        embedding = transforms.CompositeTransform([
            flows.MultiscaleNSF(
                input_shape=(3, 64, 64),
                out_shape=(8, 8),
                hidden_channels=64,
                num_levels=4 if deeper else 3,
                num_layers_per_level=3,
                num_blocks_per_layer=3,
                presqueeze=(not deeper),
            )._transform,
            transforms.OneByOneConvolution(192),
            transforms.Pad(192, 24)
        ])
        base_flow = (
            flows.MultiscaleNSF(
                input_shape=(24, 8, 8),
                hidden_channels=512,
                num_levels=4,
                num_layers_per_level=7,
                num_blocks_per_layer=3,
        ))
        super().__init__(embedding=embedding, distribution=base_flow, conformal=False)


class CelebACEFlow(flows.InjectiveFlow):
    def __init__(self):
        m = 1536
        channels = 3

        conf_embedding = transforms.CompositeTransform([
            transforms.ConformalScaleShift(channels, m),
            transforms.HouseholderConv(channels, kernel_size=4),

            transforms.ConformalScaleShift(channels*16, latent_dim=m),
            transforms.ConditionalOrthogonal1x1Conv(channels*16, channels*8),

            transforms.ConformalScaleShift(channels*8, latent_dim=m),
            transforms.HouseholderConv(channels*8, kernel_size=2),

            transforms.ConformalScaleShift(channels*32, latent_dim=m),
            transforms.ConditionalOrthogonal1x1Conv(channels*32, channels*32),

            transforms.ConformalScaleShift(channels*32, latent_dim=m),
            transforms.HouseholderConv(channels*32, kernel_size=1),

            transforms.ConformalScaleShift(channels*32, latent_dim=m),
            transforms.Orthogonal1x1Conv(channels*32, channels*8),
        ])
        base_flow = (
            flows.MultiscaleNSF(
                input_shape=(24, 8, 8),
                hidden_channels=512,
                num_levels=4,
                num_layers_per_level=7,
                num_blocks_per_layer=3,
        ))

        super().__init__(embedding=conf_embedding, distribution=base_flow, conformal=True)
