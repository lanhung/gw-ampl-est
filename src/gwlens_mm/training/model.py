"""Optional PyTorch/nflows implementation of the frozen probe architecture."""

from __future__ import annotations

import importlib
from typing import Any, Mapping


class MissingTrainingDependency(RuntimeError):
    """Raised when the isolated training environment has not been installed."""


def _dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
        flow_module = importlib.import_module("nflows.flows.base")
        distribution_module = importlib.import_module("nflows.distributions.normal")
        transforms = importlib.import_module("nflows.transforms.base")
        autoregressive = importlib.import_module("nflows.transforms.autoregressive")
        permutations = importlib.import_module("nflows.transforms.permutations")
    except ImportError as exc:
        raise MissingTrainingDependency(
            "install the isolated phase4 training optional dependencies"
        ) from exc
    return torch, nn, flow_module, distribution_module, transforms, (autoregressive, permutations)


def build_probe_model(config: Mapping[str, Any], *, seed: int) -> Any:
    """Build the seeded mask-aware conditional rational-quadratic NSF.

    Requiring the seed here prevents callers from initializing the random flow
    permutations or neural weights before the preregistered seed is applied.
    """

    torch, nn, flow_module, distribution_module, transforms, transform_parts = _dependencies()
    from .engine import set_deterministic_training_seed

    set_deterministic_training_seed(seed)
    autoregressive, permutations = transform_parts
    architecture = config["architecture"]
    gw_config = architecture["gw_encoder"]
    em_config = architecture["em_encoder"]
    context_dimension = int(architecture["context_dimension"])

    class ResidualBlock(nn.Module):  # type: ignore[name-defined]
        def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
            super().__init__()
            kernel = int(gw_config["residual_kernel"])
            padding = kernel // 2
            groups = min(8, out_channels)
            self.main = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel, stride=stride, padding=padding),
                nn.GroupNorm(groups, out_channels),
                nn.SiLU(),
                nn.Conv1d(out_channels, out_channels, kernel, padding=padding),
                nn.GroupNorm(groups, out_channels),
            )
            self.skip = (
                nn.Identity()
                if in_channels == out_channels and stride == 1
                else nn.Conv1d(in_channels, out_channels, 1, stride=stride)
            )
            self.activation = nn.SiLU()

        def forward(self, value: Any) -> Any:
            return self.activation(self.main(value) + self.skip(value))

    class SharedGWEncoder(nn.Module):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            channels = [int(value) for value in gw_config["channels"]]
            stem_kernel = int(gw_config["stem_kernel"])
            layers = [
                nn.Conv1d(
                    1,
                    channels[0],
                    stem_kernel,
                    stride=int(gw_config["stem_stride"]),
                    padding=stem_kernel // 2,
                ),
                nn.GroupNorm(min(8, channels[0]), channels[0]),
                nn.SiLU(),
            ]
            current = channels[0]
            for stage_index, output in enumerate(channels):
                for block_index in range(int(gw_config["blocks_per_stage"])):
                    stride = 2 if stage_index > 0 and block_index == 0 else 1
                    layers.append(ResidualBlock(current, output, stride))
                    current = output
            layers.extend(
                [
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(current, int(gw_config["stream_embedding_dimension"])),
                    nn.SiLU(),
                ]
            )
            self.stream = nn.Sequential(*layers)
            identity_dimension = int(gw_config["slot_identity_embedding_dimension"])
            self.image_embedding = nn.Embedding(2, identity_dimension)
            self.detector_embedding = nn.Embedding(3, identity_dimension)
            stream_dimension = int(gw_config["stream_embedding_dimension"])
            self.output_dimension = 6 * (stream_dimension + 2 * identity_dimension + 1)

        def forward(self, strain: Any, mask: Any) -> Any:
            batch = strain.shape[0]
            streams = strain.reshape(batch * 6, 1, strain.shape[-1])
            encoded = self.stream(streams).reshape(batch, 6, -1)
            flat_mask = mask.reshape(batch, 6, 1).to(encoded.dtype)
            encoded = encoded * flat_mask
            image_index = torch.arange(2, device=strain.device).repeat_interleave(3)
            detector_index = torch.arange(3, device=strain.device).repeat(2)
            image_identity = self.image_embedding(image_index)[None].expand(batch, -1, -1)
            detector_identity = self.detector_embedding(detector_index)[None].expand(
                batch, -1, -1
            )
            slots = torch.cat((encoded, image_identity, detector_identity, flat_mask), dim=-1)
            return slots.reshape(batch, -1)

    class EMEncoder(nn.Module):  # type: ignore[name-defined]
        def __init__(self, scalar_count: int, modality_count: int) -> None:
            super().__init__()
            item_hidden = int(em_config["item_hidden_dimension"])
            set_dimension = int(em_config["set_embedding_dimension"])
            scalar_hidden = int(em_config["scalar_hidden_dimension"])
            self.item = nn.Sequential(
                nn.Linear(9, item_hidden), nn.SiLU(), nn.Linear(item_hidden, set_dimension)
            )
            scalar_input = scalar_count * 2 + modality_count
            self.scalar = nn.Sequential(
                nn.Linear(scalar_input, scalar_hidden),
                nn.SiLU(),
                nn.Linear(scalar_hidden, scalar_hidden),
                nn.SiLU(),
            )
            self.output = nn.Sequential(
                nn.Linear(set_dimension + scalar_hidden, int(em_config["output_dimension"])),
                nn.SiLU(),
            )

        def forward(self, batch: Mapping[str, Any]) -> Any:
            item_mask = batch["astrometry_mask"].unsqueeze(-1)
            set_embedding = (self.item(batch["astrometry_items"]) * item_mask).sum(dim=1)
            scalar = torch.cat(
                (
                    batch["scalar_features"],
                    batch["scalar_mask"],
                    batch["modality_mask"],
                ),
                dim=-1,
            )
            return self.output(torch.cat((set_embedding, self.scalar(scalar)), dim=-1))

    class ContextEncoder(nn.Module):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.gw = SharedGWEncoder()
            self.em = EMEncoder(scalar_count=22, modality_count=7)
            input_dimension = self.gw.output_dimension + int(em_config["output_dimension"]) + 2
            self.fusion = nn.Sequential(
                nn.Linear(input_dimension, int(architecture["fusion"]["hidden_dimension"])),
                nn.SiLU(),
                nn.Linear(int(architecture["fusion"]["hidden_dimension"]), context_dimension),
            )

        def forward(self, batch: Mapping[str, Any]) -> Any:
            return self.fusion(
                torch.cat(
                    (
                        self.gw(batch["gw_strain"], batch["detector_mask"]),
                        self.em(batch),
                        batch["lens_family_condition"],
                    ),
                    dim=-1,
                )
            )

    flow_config = architecture["flow"]
    transform_list = []
    for index in range(int(flow_config["transforms"])):
        transform_list.append(permutations.RandomPermutation(features=2))
        transform_list.append(
            autoregressive.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=2,
                hidden_features=int(flow_config["conditioner_width"]),
                context_features=context_dimension,
                num_bins=int(flow_config["spline_bins"]),
                tails="linear",
                tail_bound=float(flow_config["tail_bound"]),
                num_blocks=2,
                use_residual_blocks=True,
                random_mask=False,
                dropout_probability=0.0,
                use_batch_norm=False,
            )
        )
    flow = flow_module.Flow(
        transforms.CompositeTransform(transform_list),
        distribution_module.StandardNormal((2,)),
    )

    class ConditionalPosterior(nn.Module):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.context_encoder = ContextEncoder()
            self.flow = flow

        def encode_context(self, batch: Mapping[str, Any]) -> Any:
            return self.context_encoder(batch)

        def log_prob(self, target: Any, batch: Mapping[str, Any]) -> Any:
            return self.flow.log_prob(target, context=self.encode_context(batch))

        def sample(self, sample_count: int, batch: Mapping[str, Any]) -> Any:
            return self.sample_from_context(sample_count, self.encode_context(batch))

        def sample_from_context(self, sample_count: int, context: Any) -> Any:
            return self.flow.sample(sample_count, context=context)

        def sample_log_prob(self, samples: Any, batch: Mapping[str, Any]) -> Any:
            """Evaluate each conditional draw for joint HPD coverage diagnostics."""

            if samples.ndim != 3 or samples.shape[0] != batch["gw_strain"].shape[0]:
                raise ValueError("conditional samples must be (cases, draws, targets)")
            return self.sample_log_prob_from_context(samples, self.encode_context(batch))

        def sample_log_prob_from_context(self, samples: Any, context: Any) -> Any:
            """Evaluate conditional draws without recomputing an encoded context."""

            if samples.ndim != 3 or samples.shape[0] != context.shape[0]:
                raise ValueError("conditional samples and context have incompatible cases")
            draws = samples.shape[1]
            flat_samples = samples.reshape(samples.shape[0] * draws, samples.shape[2])
            flat_context = context.repeat_interleave(draws, dim=0)
            values = self.flow.log_prob(flat_samples, context=flat_context)
            return values.reshape(samples.shape[0], draws)

    return ConditionalPosterior()
