"""Small, strict configuration for the single supported architecture."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

from omegaconf import OmegaConf


@dataclass(frozen=True)
class ModelConfig:
    """Capacity and depth of Hierarchical Kaveh.

    Patch size four, four registers, pair multiplication in every coarse
    layer, and the absence of triangle attention are architectural constants.
    """

    node_dim: int = 768
    condition_dim: int = 256
    pair_dim: int = 64
    atom_dim: int = 128
    attention_heads: int = 12
    attention_head_dim: int = 64
    atom_heads: int = 4
    atom_head_dim: int = 32
    atom_encoder_depth: int = 1
    residue_encoder_depth: int = 4
    coarse_depth: int = 6
    residue_decoder_depth: int = 4
    atom_decoder_depth: int = 1
    atom_window_radius: int = 1
    residue_ffn_expansion: int = 4
    pair_ffn_expansion: int = 4
    atom_ffn_expansion: int = 2
    dropout: float = 0.2
    patchify_mode: str = "masked_pool"
    coarse_pair_position: str = "after_node"
    coarse_pair_transition: bool = False
    pair_rbf_bins: int = 16
    pair_distance_min: float = 0.05
    pair_distance_max: float = 22.0
    pair_geometry_mode: str = "legacy"
    pair_cross_patch_geometry: bool = False
    pair_cross_patch_extrema: bool = False
    pair_self_conditioned_geometry: bool = False
    distogram_bins: int = 64
    distogram_min: float = 2.3125
    distogram_max: float = 21.6875
    sigma_data: float = 16.0
    checkpoint_blocks: bool = False
    intermediate_distograms: bool = False
    intermediate_distogram_feedback: bool = False
    secondary_structure_conditioning: bool = False
    secondary_structure_prediction: bool = False
    secondary_structure_self_conditioning: bool = False
    secondary_structure_self_conditioning_alpha: float = 0.5

    def __post_init__(self) -> None:
        positive = (
            "node_dim",
            "condition_dim",
            "pair_dim",
            "atom_dim",
            "attention_heads",
            "attention_head_dim",
            "atom_heads",
            "atom_head_dim",
            "coarse_depth",
            "residue_ffn_expansion",
            "pair_ffn_expansion",
            "atom_ffn_expansion",
            "pair_rbf_bins",
            "distogram_bins",
        )
        if any(int(getattr(self, name)) <= 0 for name in positive):
            raise ValueError("model widths, coarse depth, and expansion factors must be positive")
        depths = (
            self.atom_encoder_depth,
            self.residue_encoder_depth,
            self.residue_decoder_depth,
            self.atom_decoder_depth,
            self.atom_window_radius,
        )
        if any(int(value) < 0 for value in depths):
            raise ValueError("model depths and atom_window_radius must be non-negative")
        if self.atom_encoder_depth + self.atom_decoder_depth == 0:
            raise ValueError("at least one atom block is required")
        if self.attention_heads * self.attention_head_dim != self.node_dim:
            raise ValueError("attention_heads * attention_head_dim must equal node_dim")
        if self.atom_heads * self.atom_head_dim != self.atom_dim:
            raise ValueError("atom_heads * atom_head_dim must equal atom_dim")
        if self.patchify_mode not in {"masked_pool", "flat_linear"}:
            raise ValueError("patchify_mode must be 'masked_pool' or 'flat_linear'")
        if self.coarse_pair_position not in {"after_node", "before_attention"}:
            raise ValueError(
                "coarse_pair_position must be 'after_node' or 'before_attention'"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 < self.pair_distance_min < self.pair_distance_max:
            raise ValueError("pair distance bounds must be increasing and positive")
        if self.pair_geometry_mode not in {"legacy", "local_center"}:
            raise ValueError("pair_geometry_mode must be 'legacy' or 'local_center'")
        if self.pair_cross_patch_geometry and self.pair_geometry_mode != "local_center":
            raise ValueError("pair_cross_patch_geometry requires local_center geometry")
        if self.pair_cross_patch_extrema and not self.pair_cross_patch_geometry:
            raise ValueError("pair_cross_patch_extrema requires pair_cross_patch_geometry")
        if self.pair_self_conditioned_geometry and self.pair_geometry_mode != "local_center":
            raise ValueError("pair_self_conditioned_geometry requires local_center geometry")
        if self.intermediate_distogram_feedback and not self.intermediate_distograms:
            raise ValueError("intermediate distogram feedback requires intermediate_distograms")
        if self.secondary_structure_prediction and not self.secondary_structure_conditioning:
            raise ValueError("secondary-structure prediction requires conditioning")
        if self.secondary_structure_self_conditioning and not self.secondary_structure_prediction:
            raise ValueError("secondary-structure self-conditioning requires prediction")
        if not 0.0 <= self.secondary_structure_self_conditioning_alpha <= 1.0:
            raise ValueError("secondary-structure self-conditioning alpha must lie in [0, 1]")


@dataclass(frozen=True)
class DataConfig:
    """Existing Ragged Atom14 shards and DataLoader settings."""

    metadata_path: str = ""
    max_length: int = 128
    min_length: int = 4
    mean_plddt_min: float | None = None
    loop_length_max: int | None = None
    loop_content_max: float | None = None
    packing_density_min: float | None = None
    batch_size: int = 32
    num_workers: int = 16
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 1
    # None means that each global (rank, worker) owner preloads all of its
    # disjoint shards. An integer retains a bounded lazy cache for small-memory
    # environments and tests.
    shard_cache_size: int | None = None
    secondary_structure: bool = False
    secondary_structure_mask_probability: float = 0.5
    seed: int = 42
    length_buckets: tuple[int, ...] = (64, 96, 128)
    # One calibrated coarse capacity per effective length bucket. Without an
    # explicit calibration, use the compact single-segment capacity and reject
    # discontinuous overflows eagerly instead of recompiling the model.
    patch_capacities: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.min_length <= 0 or self.max_length < self.min_length:
            raise ValueError("data length bounds are invalid")
        if self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative")
        if self.shard_cache_size is not None and self.shard_cache_size <= 0:
            raise ValueError("shard_cache_size must be positive or null")
        if not 0.0 <= self.secondary_structure_mask_probability <= 1.0:
            raise ValueError("secondary_structure_mask_probability must lie in [0, 1]")
        if self.mean_plddt_min is not None and not 0.0 <= self.mean_plddt_min <= 100.0:
            raise ValueError("data.mean_plddt_min must lie in [0, 100]")
        if self.loop_length_max is not None and self.loop_length_max < 0:
            raise ValueError("data.loop_length_max must be nonnegative")
        if self.loop_content_max is not None and not 0.0 <= self.loop_content_max <= 1.0:
            raise ValueError("data.loop_content_max must lie in [0, 1]")
        if self.packing_density_min is not None and not 0.0 <= self.packing_density_min <= 1.0:
            raise ValueError("data.packing_density_min must lie in [0, 1]")
        if any(a >= b for a, b in zip(self.length_buckets, self.length_buckets[1:])):
            raise ValueError("length_buckets must be strictly increasing")
        if self.patch_capacities is not None:
            edges = self.effective_length_buckets
            if len(self.patch_capacities) != len(edges):
                raise ValueError(
                    "data.patch_capacities must contain one value per effective length bucket"
                )
            for capacity, edge in zip(self.patch_capacities, edges, strict=True):
                if (
                    isinstance(capacity, bool)
                    or not isinstance(capacity, int)
                    or capacity < (edge + 3) // 4
                    or capacity % 4
                ):
                    raise ValueError(
                        "data.patch_capacities must be multiples of four covering each bucket"
                    )

    @property
    def effective_length_buckets(self) -> tuple[int, ...]:
        """Return configured buckets with ``max_length`` represented once."""

        edges = tuple(edge for edge in self.length_buckets if edge <= self.max_length)
        return edges if edges and edges[-1] == self.max_length else (*edges, self.max_length)


@dataclass(frozen=True)
class DiffusionConfig:
    """Pallatom's log-normal EDM training distribution."""

    p_mean: float = -1.2
    p_std: float = 1.5
    translation_std: float = 1.0

    def __post_init__(self) -> None:
        if self.p_std <= 0 or self.translation_std < 0:
            raise ValueError("diffusion p_std must be positive and translation_std non-negative")


@dataclass(frozen=True)
class LossConfig:
    """Published Pallatom terminal losses supported by this architecture."""

    coordinate_weight: float = 1.0
    align_coordinate_loss: bool = True
    aatype_weight: float = 0.25
    aatype_sigma_max: float = 0.5
    aatype_sigma_ramp_max: float | None = None
    aatype_marginal_js_weight: float = 0.0
    smooth_lddt_weight: float = 1.0
    distogram_weight: float = 0.5
    intermediate_distogram_weight: float = 0.0
    secondary_structure_weight: float = 0.0
    polar_aatypes: str = "RNDCEQHKSTY"
    polar_weight: float = 2.0
    smooth_lddt_cutoff: float = 15.0
    smooth_lddt_chunk_size: int = 128
    distogram_drop_diagonal: bool = False

    def __post_init__(self) -> None:
        alphabet = set("ARNDCQEGHILKMFPSTWYV")
        if not self.polar_aatypes or len(set(self.polar_aatypes)) != len(self.polar_aatypes):
            raise ValueError("loss.polar_aatypes must contain unique residue codes")
        if not set(self.polar_aatypes) <= alphabet:
            raise ValueError("loss.polar_aatypes contains an unknown residue code")
        if self.aatype_sigma_max <= 0:
            raise ValueError("loss.aatype_sigma_max must be positive")
        if (
            self.aatype_sigma_ramp_max is not None
            and self.aatype_sigma_ramp_max <= self.aatype_sigma_max
        ):
            raise ValueError(
                "loss.aatype_sigma_ramp_max must exceed aatype_sigma_max"
            )
        if self.aatype_marginal_js_weight < 0:
            raise ValueError("loss.aatype_marginal_js_weight must be non-negative")
        if self.polar_weight <= 0 or self.smooth_lddt_cutoff <= 0:
            raise ValueError("loss polar weight and lDDT cutoff must be positive")
        if self.smooth_lddt_chunk_size <= 0:
            raise ValueError("loss.smooth_lddt_chunk_size must be positive")
        if self.intermediate_distogram_weight < 0:
            raise ValueError("loss intermediate distogram weight must be non-negative")
        if self.secondary_structure_weight < 0:
            raise ValueError("secondary_structure_weight must be non-negative")


@dataclass(frozen=True)
class EmaConfig:
    enabled: bool = True
    profile: str = "constant"
    decay: float = 0.999
    srel: float = 0.1
    offload_to_cpu: bool = True
    update_every: int = 2
    compensate_update_every: bool = True


@dataclass(frozen=True)
class CompileConfig:
    """Whole-model compile policy consumed directly by Koochak."""

    enabled: bool = False
    preset: str | None = None
    mode: str = "default"
    fullgraph: bool = False
    dynamic: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    """Koochak loop settings for Pallatom's 300k-step training run."""

    max_steps: int = 300_000
    log_every: int = 5
    eval_every: int = 1_000_000_000
    eval_at_step_zero: bool = False
    ckpt_every: int = 10_000
    save_final: bool = True
    grad_accum: int = 1
    grad_clip_norm: float = 1.0
    nonfinite_grad_check_every: int = 100
    scheduler_step: str = "step"
    amp: str = "bf16"
    ddp: bool = True
    ddp_static_graph: bool = False
    ddp_gradient_as_bucket_view: bool = True
    ddp_bucket_cap_mb: int = 32
    ddp_broadcast_buffers: bool = False
    # Evacuation is deliberately opt-in.  Koochak installs the USR1 handler
    # only when this flag is enabled, leaving ordinary local training intact.
    evacuation_enabled: bool = False
    evacuation_signal: str = "USR1"
    seed: int = 42
    device: str = "cuda"
    out_dir: str = "./runs/hierarchical-kaveh"
    keep_last_k: int = 3
    prefetch_batches: int = 2
    prefetch_threaded: bool = True
    prefetch_pipeline: str = "two_stage"
    autocast_in_step_fn: bool = True
    scalarize_loss_every_step: bool = False
    profile_step_fn_timing: bool = False
    profile_step_fn_cuda_sync: bool = True
    self_conditioning_probability: float = 1.0
    compile: CompileConfig = field(default_factory=CompileConfig)
    require_compile: bool = False
    require_fused: bool = False
    ema: EmaConfig = field(default_factory=EmaConfig)

    def __post_init__(self) -> None:
        if not 0.0 <= self.self_conditioning_probability <= 1.0:
            raise ValueError("train.self_conditioning_probability must lie in [0, 1]")
        if self.evacuation_signal != "USR1":
            raise ValueError("train.evacuation_signal must equal 'USR1'")


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass(frozen=True)
class LoggingConfig:
    csv_path: str | None = None
    jsonl_path: str | None = None


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = False
    project: str | None = None
    entity: str | None = None
    name: str | None = None
    id: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()
    mode: str | None = None
    resume: str | None = None


@dataclass(frozen=True)
class SamplingConfig:
    """Published Pallatom stochastic Euler sampler."""

    num_steps: int = 200
    p_mean: float = -1.2
    p_std: float = 1.5
    gamma: float = 0.2
    churn_tmin: float = 0.01
    churn_tmax: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 2.25
    translation_std: float = 1.0
    sequence_temperature: float = 0.1

    def __post_init__(self) -> None:
        if self.num_steps <= 0 or self.p_std <= 0:
            raise ValueError("sampling num_steps and p_std must be positive")
        if not 0 <= self.churn_tmin < self.churn_tmax <= 1:
            raise ValueError("sampling churn interval must lie within [0,1]")
        if self.gamma < 0 or self.noise_scale <= 0 or self.step_scale <= 0:
            raise ValueError("sampling stochastic and step scales must be positive")
        if self.translation_std < 0 or self.sequence_temperature <= 0:
            raise ValueError("sampling translation std and sequence temperature are invalid")


@dataclass(frozen=True)
class RunConfig:
    """Complete training and sampling configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return plain values suitable for Koochak and checkpoint metadata."""

        return asdict(self)


ConfigT = TypeVar("ConfigT")


def _strict_construct(cls: type[ConfigT], values: Mapping[str, Any]) -> ConfigT:
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**dict(values))


def load_config(file: str | Path) -> RunConfig:
    """Load a strict YAML configuration over the compact defaults."""

    raw = OmegaConf.to_container(OmegaConf.load(file), resolve=True)
    if not isinstance(raw, Mapping):
        raise ValueError("configuration root must be a mapping")
    allowed = {item.name for item in fields(RunConfig)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown RunConfig sections: {sorted(unknown)}")

    def section(name: str, cls: type[ConfigT]) -> ConfigT:
        values = raw.get(name, {})
        if not isinstance(values, Mapping):
            raise ValueError(f"configuration section {name!r} must be a mapping")
        normalized = dict(values)
        for key in ("length_buckets", "patch_capacities", "betas", "tags"):
            if key in normalized and isinstance(normalized[key], list):
                normalized[key] = tuple(normalized[key])
        return _strict_construct(cls, normalized)

    train_values = raw.get("train", {})
    if not isinstance(train_values, Mapping):
        raise ValueError("configuration section 'train' must be a mapping")
    train_values = dict(train_values)
    ema_values = train_values.pop("ema", {})
    if not isinstance(ema_values, Mapping):
        raise ValueError("configuration section 'train.ema' must be a mapping")
    compile_values = train_values.pop("compile", {})
    if not isinstance(compile_values, Mapping):
        raise ValueError("configuration section 'train.compile' must be a mapping")
    train = _strict_construct(
        TrainingConfig,
        {
            **train_values,
            "compile": _strict_construct(CompileConfig, compile_values),
            "ema": _strict_construct(EmaConfig, ema_values),
        },
    )
    config = RunConfig(
        model=section("model", ModelConfig),
        data=section("data", DataConfig),
        diffusion=section("diffusion", DiffusionConfig),
        loss=section("loss", LossConfig),
        train=train,
        optimizer=section("optimizer", OptimizerConfig),
        logging=section("logging", LoggingConfig),
        wandb=section("wandb", WandbConfig),
        sampling=section("sampling", SamplingConfig),
    )
    if config.data.metadata_path == "":
        raise ValueError("data.metadata_path is required")
    if config.train.require_compile and not config.train.compile.enabled:
        raise ValueError("train.require_compile requires train.compile.enabled")
    if config.train.require_fused and config.train.device != "cuda":
        raise ValueError("train.require_fused requires train.device=cuda")
    return config
