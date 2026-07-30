import torch
import torch.nn as nn
from ..layers import OrPooling2d, GroupSum, LogicConv2d, LogicDense
from ..layers.binarization import setup_binarization
from ..topology import (
    add_identity_to_ancestry,
    combine_channel_spatial_ancestry,
    image_input_semantics,
    packed_identity_in_universe,
)


class CNN(torch.nn.Module):
    """An implementation of a logic gate convolutional neural network."""

    def __init__(self, class_count, tau, parametrization="raw", **llkw):
        super(CNN, self).__init__()
        logic_layers = []
        # specifically written for mnist
        k_num = 16
        logic_layers.append(
            LogicConv2d(
                in_dim=28,
                num_kernels=k_num,
                channels=1,
                **llkw,
                tree_depth=3,
                receptive_field_size=5,
                parametrization=parametrization,
                padding=0,
            )
        )
        logic_layers.append(OrPooling2d(kernel_size=2, stride=2, padding=0))

        logic_layers.append(
            LogicConv2d(
                in_dim=12,
                channels=k_num,
                num_kernels=3 * k_num,
                **llkw,
                tree_depth=3,
                receptive_field_size=3,
                padding=0,
                parametrization=parametrization,
            )
        )
        logic_layers.append(OrPooling2d(kernel_size=2, stride=2, padding=1))

        logic_layers.append(
            LogicConv2d(
                in_dim=6,
                channels=3 * k_num,
                num_kernels=9 * k_num,
                **llkw,
                tree_depth=3,
                receptive_field_size=3,
                padding=0,
                parametrization=parametrization,
            )
        )
        logic_layers.append(OrPooling2d(kernel_size=2, stride=2, padding=1))

        logic_layers.append(torch.nn.Flatten())

        logic_layers.append(LogicDense(in_dim=81 * k_num, out_dim=1280 * k_num, parametrization=parametrization, **llkw))
        logic_layers.append(LogicDense(in_dim=1280 * k_num, out_dim=640 * k_num, parametrization=parametrization, **llkw))
        logic_layers.append(LogicDense(in_dim=640 * k_num, out_dim=320 * k_num, parametrization=parametrization, **llkw))

        self.model = torch.nn.Sequential(*logic_layers, GroupSum(class_count, tau))

    def forward(self, x):
        """Forward pass of the logic gate convolutional neural network."""
        return self.model(x)


class ClgnMnist(torch.nn.Sequential):
    """
    Model as described in the paper 'Convolutional Logic Gate Networks'
    for the MNIST dataset.
    """

    def __init__(self, thresholds: torch.Tensor, binarization: str, binarization_kwargs: dict, 
                 k_num: int=16, parametrization="raw", tau=1.0, **llkw):
        
        binarization = "dummy"
        binarization_module = setup_binarization(thresholds, binarization, **binarization_kwargs)
        self.k_num = k_num
        layers = [binarization_module]
        layers.append(
            LogicConv2d(
                in_dim=28,
                num_kernels=k_num,
                channels=1,
                **llkw,
                tree_depth=3,
                receptive_field_size=5,
                padding=0,
                parametrization=parametrization,
            )
        )
        layers.append(OrPooling2d(kernel_size=2, stride=2, padding=0))

        layers.append(
            LogicConv2d(
                in_dim=12,
                channels=k_num,
                num_kernels=3 * k_num,
                **llkw,
                tree_depth=3,
                receptive_field_size=3,
                padding=0,
                parametrization=parametrization,
            )
        )
        layers.append(OrPooling2d(kernel_size=2, stride=2, padding=1))

        layers.append(
            LogicConv2d(
                in_dim=6,
                channels=3 * k_num,
                num_kernels=9 * k_num,
                **llkw,
                tree_depth=3,
                receptive_field_size=3,
                padding=0,
                parametrization=parametrization,
            )
        )
        layers.append(OrPooling2d(kernel_size=2, stride=2, padding=1))

        layers.append(torch.nn.Flatten())

        layers.append(LogicDense(in_dim=81 * k_num, out_dim=1280 * k_num, parametrization=parametrization, **llkw))
        layers.append(LogicDense(in_dim=1280 * k_num, out_dim=640 * k_num, parametrization=parametrization, **llkw))
        layers.append(LogicDense(in_dim=640 * k_num, out_dim=320 * k_num, parametrization=parametrization, **llkw))

        super(ClgnMnist, self).__init__(*layers, GroupSum(k=10, tau=tau))


class ClgnMnistTiny(ClgnMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 1.0)
        super(ClgnMnistTiny, self).__init__(k_num=4, tau=tau, **llkw)


class ClgnMnistSmall(ClgnMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 6.5)
        super(ClgnMnistSmall, self).__init__(k_num=16, tau=tau, **llkw)


class ClgnMnistMedium(ClgnMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 28.)
        super(ClgnMnistMedium, self).__init__(k_num=64, tau=tau, **llkw)


class ClgnMnistLarge(ClgnMnist):
    def __init__(self, **llkw):
        tau = llkw.get("tau", 35.)
        super(ClgnMnistLarge, self).__init__(k_num=1024, tau=tau, **llkw)


class ClgnCifar10(torch.nn.Sequential):
    """
    An implementation of a logic gate convolutional neural network for CIFAR-10,
    as described in the paper 'convolutional logic gate networks'.
    Provided in three sizes: small, medium, large.
    Small and medium take 2-bit-thresholded inputs, large takes 5-bit-thresholded inputs. 
    """
    n_input_bits = None
    # ``n_input_bits`` is retained for compatibility with existing TorchLogix
    # classes and configurations.  The convolutional paper distinguishes input
    # precision from the number of Boolean thermometer channels: its 2-bit S/M
    # input uses three thresholds.  New paper-faithful classes therefore set
    # ``n_input_thresholds`` explicitly.
    n_input_thresholds = None
    input_precision_bits = None
    k_num = None
    tau = None
    group_size = None
    group_size_input = None
    output_gate_factor = 1
    paper_model_identifier = None

    def __init__(self, thresholds: torch.Tensor, binarization: str, binarization_kwargs: dict, connections_kwargs: dict, **llkw):
        threshold_count = (
            self.n_input_thresholds
            if self.n_input_thresholds is not None
            else self.n_input_bits
        )
        assert thresholds.shape[-1] == threshold_count, (
            f"{self.__class__.__name__} requires {threshold_count} input "
            f"thresholds, got {thresholds.shape[-1]}."
        )
        binarization_kwargs = dict(binarization_kwargs)  # make a copy to avoid modifying the original
        binarization_kwargs["feature_dim"] = 1  # image data
        n_thresholds = thresholds.shape[-1]
        binarization_module = setup_binarization(thresholds, binarization, **binarization_kwargs)

        base_connections_kwargs = dict(connections_kwargs)
        conv_method = (
            base_connections_kwargs.pop("conv_init_method", None)
            or base_connections_kwargs["init_method"]
        )
        classifier_method = base_connections_kwargs.pop(
            "classifier_init_method", None
        )
        conv_connections_kwargs = dict(base_connections_kwargs)
        conv_connections_kwargs["init_method"] = conv_method
        conv_connections_kwargs["channel_group_size"] = self.group_size
        dense_connections_kwargs = dict(base_connections_kwargs)

        group_size_input = (
            self.group_size_input
            if self.group_size_input is not None
            else self.group_size
        )

        layers = [binarization_module]
        input_channels = 3 * n_thresholds
        conv_specs = [
            (32, input_channels, self.k_num, group_size_input),
            (16, self.k_num, 4 * self.k_num, self.group_size),
            (8, 4 * self.k_num, 16 * self.k_num, self.group_size),
            (4, 16 * self.k_num, 32 * self.k_num, self.group_size),
        ]
        track_channel_ancestry = (
            conv_method in {
                "semantic_channel_hybrid",
                "semantic_channel_spatial_hybrid",
                "ancestry_channel_hybrid",
                "coverage_reuse_hybrid",
            }
            and (
                conv_method in {
                    "ancestry_channel_hybrid",
                    "coverage_reuse_hybrid",
                }
                or classifier_method in {
                    "semantic_balanced_hybrid",
                    "semantic_classifier_hybrid",
                }
            )
        )
        channel_ancestry = None
        ancestry_offset = input_channels
        ancestry_universe = input_channels + sum(
            out_channels for _, _, out_channels, _ in conv_specs
        )
        if track_channel_ancestry:
            channel_ancestry = packed_identity_in_universe(
                input_channels,
                ancestry_universe,
            )

        for layer_index, (
            spatial_dim,
            in_channels,
            out_channels,
            channel_group_size,
        ) in enumerate(conv_specs):
            layer_connections_kwargs = conv_connections_kwargs | {
                "channel_group_size": channel_group_size,
                "layer_index": layer_index,
            }
            if channel_ancestry is not None:
                layer_connections_kwargs["input_channel_ancestry"] = (
                    channel_ancestry
                )
            if layer_index == 0:
                layer_connections_kwargs["semantic_threshold_count"] = (
                    n_thresholds
                )
            conv_layer = LogicConv2d(
                in_dim=spatial_dim,
                channels=in_channels,
                num_kernels=out_channels,
                tree_depth=3,
                receptive_field_size=3,
                padding=1,
                connections_kwargs=layer_connections_kwargs,
                **llkw,
            )
            layers.append(conv_layer)
            layers.append(OrPooling2d(kernel_size=2, stride=2))
            if channel_ancestry is not None:
                channel_ancestry = (
                    conv_layer.connections.consume_output_channel_ancestry()
                )
                if channel_ancestry is None:
                    raise RuntimeError(
                        f"{conv_method} did not return convolutional ancestry"
                    )
                channel_ancestry = add_identity_to_ancestry(
                    channel_ancestry,
                    offset=ancestry_offset,
                    universe_size=ancestry_universe,
                )
                ancestry_offset += out_channels

        layers.append(torch.nn.Flatten())

        dense_specs = [
            (128 * self.k_num, 1280 * self.k_num),
            (1280 * self.k_num, 640 * self.k_num),
            (
                640 * self.k_num,
                320 * self.k_num * self.output_gate_factor,
            ),
        ]
        dense_ancestry = None
        dense_semantics = None
        if channel_ancestry is not None:
            dense_ancestry = combine_channel_spatial_ancestry(
                channel_ancestry,
                spatial_positions=4,
            )
            dense_semantics = image_input_semantics(
                channels=32 * self.k_num,
                height=2,
                width=2,
                threshold_bits=1,
                layout="channel_interleaved",
            )

        for dense_index, (in_dim, out_dim) in enumerate(dense_specs):
            layer_kwargs = {}
            if classifier_method is not None:
                layer_connections_kwargs = dense_connections_kwargs | {
                    "init_method": classifier_method,
                    "layer_index": 4 + dense_index,
                    "output_groups": 10 if dense_index == 2 else 1,
                }
                if dense_ancestry is not None:
                    layer_connections_kwargs["input_ancestry"] = dense_ancestry
                if dense_index == 0 and dense_semantics is not None:
                    layer_connections_kwargs["input_semantics"] = dense_semantics
                layer_kwargs["connections_kwargs"] = layer_connections_kwargs
            dense_layer = LogicDense(
                in_dim=in_dim,
                out_dim=out_dim,
                **layer_kwargs,
                **llkw,
            )
            layers.append(dense_layer)
            if classifier_method is not None:
                dense_ancestry = (
                    dense_layer.connections.consume_output_ancestry()
                )

        super(ClgnCifar10, self).__init__(*layers, GroupSum(k=10, tau=self.tau))



class ClgnCifar10Small(ClgnCifar10):
    n_input_bits = 2
    k_num = 32
    tau = 20
    group_size = 2


class ClgnCifar10Medium(ClgnCifar10):
    n_input_bits = 2
    k_num = 256
    tau = 40
    group_size = 2


class ClgnCifar10Large(ClgnCifar10):
    n_input_bits = 5
    k_num = 512
    tau = 280
    group_size = 2


class ClgnCifar10PaperSmall(ClgnCifar10):
    """LogicTreeNet-S from Petersen et al. (NeurIPS 2024).

    The paper uses 2-bit RGB input precision represented by three thermometer
    thresholds per channel.  This is deliberately separate from the legacy
    :class:`ClgnCifar10Small`, whose two thresholds are retained so existing
    checkpoints and CoverageDLGN v4 pilot results remain reproducible.
    """

    n_input_bits = 2
    n_input_thresholds = 3
    input_precision_bits = 2
    k_num = 32
    tau = 20
    group_size = 2
    output_gate_factor = 1
    paper_model_identifier = "S"


class ClgnCifar10PaperMedium(ClgnCifar10):
    """LogicTreeNet-M from Petersen et al. (NeurIPS 2024)."""

    n_input_bits = 2
    n_input_thresholds = 3
    input_precision_bits = 2
    k_num = 256
    tau = 40
    group_size = 2
    output_gate_factor = 1
    paper_model_identifier = "M"


class ClgnCifar10Small2(ClgnCifar10):
    n_input_bits = 2
    k_num = 32
    tau = 20
    group_size = 2
    group_size_input = 1


class ClgnCifar10Medium2(ClgnCifar10):
    n_input_bits = 2
    k_num = 256
    tau = 40
    group_size = 2
    group_size_input = 1
