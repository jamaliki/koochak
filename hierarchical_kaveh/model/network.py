"""The complete atom -> residue -> p=4 pair -> residue -> atom denoiser."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ModelConfig
from ..types import DenoiserInput, Prediction
from .attention import (
    AtomBlock,
    AtomInput,
    AtomOutput,
    AtomToResidue,
    GlobalBlock,
    build_packed_layout,
)
from .layers import FeedForward, TimeEmbedding, checkpoint, init_linear
from .pair import (
    CoarseBlock,
    DistogramHead,
    DistogramPairFeedback,
    IntermediateDistogramHead,
    PairInitializer,
)
from .patch import Patchify, Unpatchify, build_patch_layout


REGISTER_COUNT = 4
AA_INPUT_CLASSES = 22  # 20 standard amino acids, unknown/mask, padding
AA_OUTPUT_CLASSES = 20
UNKNOWN_AA = 20
PAD_AA = 21
SS_INPUT_CLASSES = 5  # H, E, L, unknown/mask, padding
SS_OUTPUT_CLASSES = 3
UNKNOWN_SS = 3
PAD_SS = 4


def _broadcast_sigma(sigma: Tensor, coordinates: Tensor) -> Tensor:
    """Normalize scalar/per-example/per-residue sigma to `[B,N,14]`."""

    batch, residues, atoms = coordinates.shape[:3]
    if sigma.ndim == 0:
        sigma = sigma.expand(batch, residues, atoms)
    elif sigma.shape == (batch,):
        sigma = sigma[:, None, None].expand(-1, residues, atoms)
    elif sigma.shape == (batch, residues):
        sigma = sigma[..., None].expand(-1, -1, atoms)
    elif sigma.shape != (batch, residues, atoms):
        raise ValueError(
            f"sigma must be scalar, [B], [B,N], or [B,N,14], got {tuple(sigma.shape)}"
        )
    return sigma.to(device=coordinates.device, dtype=coordinates.dtype)


class ResidueInput(nn.Module):
    """Masked residue metadata, optionally including secondary structure."""

    def __init__(
        self,
        node_dim: int,
        *,
        secondary_structure_conditioning: bool = False,
        secondary_structure_self_conditioning: bool = False,
        secondary_structure_self_conditioning_alpha: float = 0.5,
    ):
        super().__init__()
        self.chain_break = nn.Linear(1, node_dim, bias=False)
        self.aatype = nn.Linear(AA_INPUT_CLASSES, node_dim, bias=False)
        # A one-hot selects one column: initialize these as embedding tables.
        nn.init.normal_(self.aatype.weight, std=1.0)
        self.secondary_structure_conditioning = bool(secondary_structure_conditioning)
        self.secondary_structure_self_conditioning = bool(secondary_structure_self_conditioning)
        if self.secondary_structure_conditioning:
            self.secondary_structure = nn.Linear(SS_INPUT_CLASSES, node_dim, bias=False)
            nn.init.normal_(self.secondary_structure.weight, std=1.0)
        if self.secondary_structure_self_conditioning:
            self.secondary_structure_self = nn.Linear(SS_OUTPUT_CLASSES, node_dim, bias=False)
            nn.init.zeros_(self.secondary_structure_self.weight)
        self.secondary_structure_self_conditioning_alpha = float(
            secondary_structure_self_conditioning_alpha
        )
        piece_count = 2 + int(self.secondary_structure_conditioning)
        self.scales = nn.Parameter(torch.full((piece_count,), 1.0 / math.sqrt(piece_count)))

    @staticmethod
    def _indices(value: Tensor | None, residue_mask: Tensor) -> Tensor:
        if value is None:
            value = torch.full_like(residue_mask, UNKNOWN_AA, dtype=torch.long)
        else:
            if value.shape != residue_mask.shape:
                raise ValueError("amino-acid input must have [B,N] shape")
            value = value.long().clamp(0, UNKNOWN_AA)
        return torch.where(residue_mask, value, torch.full_like(value, PAD_AA))

    @staticmethod
    def _secondary_structure_indices(value: Tensor | None, residue_mask: Tensor) -> Tensor:
        if value is None:
            value = torch.full_like(residue_mask, UNKNOWN_SS, dtype=torch.long)
        else:
            if value.shape != residue_mask.shape:
                raise ValueError("secondary-structure input must have [B,N] shape")
            value = value.long().clamp(0, UNKNOWN_SS)
        return torch.where(residue_mask, value, torch.full_like(value, PAD_SS))

    def forward(
        self,
        aatype_input: Tensor | None,
        chain_break: Tensor,
        residue_mask: Tensor,
        secondary_structure_input: Tensor | None = None,
        self_conditioned_secondary_structure: Tensor | None = None,
    ) -> Tensor:
        aa = F.one_hot(
            self._indices(aatype_input, residue_mask), AA_INPUT_CLASSES
        ).to(self.aatype.weight.dtype)
        pieces = [
            self.chain_break(chain_break[..., None].to(self.chain_break.weight.dtype)),
            self.aatype(aa),
        ]
        if self.secondary_structure_conditioning:
            ss = F.one_hot(
                self._secondary_structure_indices(secondary_structure_input, residue_mask),
                SS_INPUT_CLASSES,
            ).to(self.secondary_structure.weight.dtype)
            pieces.append(self.secondary_structure(ss))
            if self.secondary_structure_self_conditioning:
                if self_conditioned_secondary_structure is None:
                    ss_self = torch.zeros_like(pieces[-1])
                else:
                    if self_conditioned_secondary_structure.shape != (*residue_mask.shape, SS_OUTPUT_CLASSES):
                        raise ValueError(
                            "self-conditioned secondary structure must have [B,N,3] shape"
                        )
                    probabilities = self_conditioned_secondary_structure.float().softmax(-1)
                    predicted = self.secondary_structure_self(probabilities.to(pieces[-1].dtype))
                    alpha = self.secondary_structure_self_conditioning_alpha
                    pieces[-1] = (1.0 - alpha) * pieces[-1] + alpha * predicted
        stacked = torch.stack(pieces, dim=-2)
        x = (stacked * self.scales.to(stacked.dtype)[..., None]).sum(-2)
        return x * residue_mask[..., None].to(x.dtype)


class SecondaryStructureHead(nn.Module):
    """Three-class H/E/L predictor over final residue representations."""

    def __init__(self, node_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(node_dim)
        self.features = nn.Linear(node_dim, node_dim)
        self.output = init_linear(nn.Linear(node_dim, SS_OUTPUT_CLASSES, bias=False), "zero")

    def forward(self, residues: Tensor, residue_mask: Tensor) -> Tensor:
        logits = self.output(F.relu(self.features(self.norm(residues))))
        return logits * residue_mask[..., None].to(logits.dtype)


class AtomSequenceHead(nn.Module):
    """Pallatom sequence head over final per-atom representations."""

    def __init__(self, atom_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(atom_dim)
        self.features = nn.Linear(atom_dim, atom_dim)
        self.output = init_linear(
            nn.Linear(atom_dim, AA_OUTPUT_CLASSES, bias=False),
            "zero",
        )

    def forward(self, atoms: Tensor, atom_mask: Tensor, residue_count: int) -> Tensor:
        features = F.relu(self.features(self.norm(atoms)))
        mask = atom_mask[..., None].to(features.dtype)
        residue = (features * mask).sum(2) / mask.sum(2).clamp_min(1.0)
        logits = self.output(residue)
        return F.pad(logits, (0, 0, 0, residue_count - logits.shape[1]))


class HierarchicalKaveh(nn.Module):
    """One explicit, stable H6 architecture without legacy execution paths."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        c = config
        trunk_depth = c.residue_encoder_depth + c.coarse_depth + c.residue_decoder_depth
        residual_scale = 1.0 / math.sqrt(2.0 * trunk_depth)

        self.time_embedding = TimeEmbedding(14, c.condition_dim)
        self.time_ffn = FeedForward(c.condition_dim, None, 2, 0.0)
        self.residue_input = ResidueInput(
            c.node_dim,
            secondary_structure_conditioning=c.secondary_structure_conditioning,
            secondary_structure_self_conditioning=c.secondary_structure_self_conditioning,
            secondary_structure_self_conditioning_alpha=c.secondary_structure_self_conditioning_alpha,
        )
        self.registers = nn.Parameter(torch.randn(REGISTER_COUNT, c.node_dim) / 20.0)

        self.atom_input = AtomInput(c.node_dim, c.condition_dim, c.atom_dim)
        atom_args = (
            c.atom_dim, c.condition_dim, c.atom_heads, c.atom_head_dim,
            c.atom_window_radius, c.atom_ffn_expansion, c.dropout,
        )
        self.atom_encoder = nn.ModuleList(AtomBlock(*atom_args) for _ in range(c.atom_encoder_depth))
        self.atom_to_residue = AtomToResidue(c.atom_dim, c.node_dim, c.condition_dim)

        global_args = (
            c.node_dim, c.condition_dim, c.attention_heads, c.attention_head_dim,
            c.residue_ffn_expansion, c.dropout, residual_scale,
        )
        self.residue_encoder = nn.ModuleList(
            GlobalBlock(*global_args) for _ in range(c.residue_encoder_depth)
        )
        self.patchify = Patchify(c.node_dim, c.condition_dim)
        self.pair_initializer = PairInitializer(
            c.pair_dim,
            c.pair_rbf_bins,
            c.pair_distance_min,
            c.pair_distance_max,
            c.pair_geometry_mode,
            c.pair_self_conditioned_geometry,
            c.pair_cross_patch_geometry,
            c.pair_cross_patch_extrema,
        )
        self.coarse = nn.ModuleList(
            CoarseBlock(
                c.node_dim, c.condition_dim, c.pair_dim,
                c.attention_heads, c.attention_head_dim,
                c.residue_ffn_expansion, c.dropout, residual_scale, REGISTER_COUNT,
            )
            for _ in range(c.coarse_depth)
        )
        self.unpatchify = Unpatchify(c.node_dim)
        self.residue_decoder = nn.ModuleList(
            GlobalBlock(*global_args) for _ in range(c.residue_decoder_depth)
        )

        self.atom_output = AtomOutput(c.node_dim, c.atom_dim)
        self.atom_decoder = nn.ModuleList(AtomBlock(*atom_args) for _ in range(c.atom_decoder_depth))
        self.aatype_output = AtomSequenceHead(c.atom_dim)
        self.distogram = DistogramHead(c.pair_dim, c.distogram_bins)
        self.intermediate_distogram = (
            IntermediateDistogramHead(c.pair_dim, c.condition_dim, c.distogram_bins)
            if c.intermediate_distograms else None
        )
        self.distogram_pair_feedback = (
            DistogramPairFeedback(c.distogram_bins, c.pair_dim)
            if c.intermediate_distogram_feedback else None
        )
        self.secondary_structure_output = (
            SecondaryStructureHead(c.node_dim)
            if c.secondary_structure_prediction else None
        )

    def _validate(self, inputs: DenoiserInput) -> tuple[Tensor, Tensor]:
        coordinates, atom_mask = inputs.coordinates, inputs.atom_mask.bool()
        if coordinates.ndim != 4 or coordinates.shape[-2:] != (14, 3):
            raise ValueError("coordinates must have [B,N,14,3] shape")
        if atom_mask.shape != coordinates.shape[:3]:
            raise ValueError("atom_mask must have [B,N,14] shape")
        residue_shape = coordinates.shape[:2]
        for name in ("residue_index", "chain_index", "chain_break"):
            if getattr(inputs, name).shape != residue_shape:
                raise ValueError(f"{name} must have [B,N] shape")
        residue_mask = inputs.residue_mask
        prefix = torch.arange(residue_shape[1], device=coordinates.device)[None] < residue_mask.sum(1)[:, None]
        if not bool(torch.all(residue_mask == prefix)):
            raise ValueError("residues must be left-aligned and prefix padded")
        if not bool(residue_mask.any()):
            raise ValueError("a batch must contain at least one valid residue")
        return atom_mask & residue_mask[..., None], residue_mask

    def _conditions(self, sigma: Tensor, residue_mask: Tensor) -> tuple[Tensor, Tensor]:
        sigma = sigma.clamp_min(1.0e-12)
        c_noise = (sigma / self.config.sigma_data).log() / 4.0
        condition = self.time_ffn(self.time_embedding(c_noise))
        condition = condition * residue_mask[..., None].to(condition.dtype)
        return sigma, condition

    def _residue_stage(
        self,
        blocks: nn.ModuleList,
        x: Tensor,
        condition: Tensor,
        mask: Tensor,
        positions: Tensor,
    ) -> Tensor:
        # Hopper uses packed varlen attention; CPU and other GPUs use exact dense SDPA.
        use_packed = x.is_cuda and x.dtype in {torch.float16, torch.bfloat16}
        layout = build_packed_layout(mask) if use_packed else None
        if layout is not None:
            packed_x = layout.pack(x)
            packed_condition = layout.pack(condition)
            packed_positions = layout.pack(positions)
            for block in blocks:
                if self.config.checkpoint_blocks and torch.is_grad_enabled():
                    packed_x = torch.utils.checkpoint.checkpoint(
                        block.forward_packed, packed_x, packed_condition, packed_positions,
                        layout.cumulative_lengths, layout.max_length, use_reentrant=False,
                    )
                else:
                    packed_x = block.forward_packed(
                        packed_x, packed_condition, packed_positions,
                        layout.cumulative_lengths, layout.max_length,
                    )
            return layout.unpack(packed_x)
        for block in blocks:
            x = checkpoint(block, x, condition, mask, positions, enabled=self.config.checkpoint_blocks)
        return x

    def forward(
        self,
        inputs: DenoiserInput,
        *,
        compute_distogram: bool = True,
        compute_intermediate_distograms: bool = False,
    ) -> Prediction:
        atom_mask, residue_mask = self._validate(inputs)
        raw_coordinates = inputs.coordinates
        sigma = _broadcast_sigma(inputs.sigma, raw_coordinates)
        sigma, residue_condition = self._conditions(sigma, residue_mask)

        mask_float = residue_mask.to(sigma.dtype)
        sigma_sq = (mask_float[..., None] * sigma.square()).sum((1, 2), keepdim=True)
        sigma_sq = sigma_sq / (14.0 * mask_float.sum(1, keepdim=True)).clamp_min(1.0).unsqueeze(-1)
        c_in = (self.config.sigma_data**2 + sigma_sq).rsqrt()
        coordinates = c_in[..., None] * raw_coordinates
        raw_self_conditioned_coordinates = inputs.self_conditioned_coordinates
        self_conditioned_coordinates = raw_self_conditioned_coordinates
        if self_conditioned_coordinates is not None:
            if self_conditioned_coordinates.shape != raw_coordinates.shape:
                raise ValueError("self_conditioned_coordinates must match coordinates")
            self_conditioned_coordinates = self_conditioned_coordinates / self.config.sigma_data

        layout = build_patch_layout(residue_mask, inputs.chain_index, inputs.residue_index)
        residue_x = self.residue_input(
            inputs.aatype_input,
            inputs.chain_break,
            residue_mask,
            inputs.secondary_structure_input,
            inputs.self_conditioned_secondary_structure,
        )
        batch_size, residue_count = residue_mask.shape
        registers = self.registers[None].expand(batch_size, -1, -1)
        register_condition = residue_condition.new_zeros(batch_size, REGISTER_COUNT, self.config.condition_dim)
        token_condition = torch.cat((register_condition, residue_condition), 1)
        token_mask = torch.cat(
            (torch.ones(batch_size, REGISTER_COUNT, dtype=torch.bool, device=residue_mask.device), residue_mask), 1
        )
        register_positions = torch.arange(1, REGISTER_COUNT + 1, device=residue_mask.device)[None].expand(batch_size, -1)
        token_positions = torch.cat((register_positions, inputs.residue_index + REGISTER_COUNT + 1), 1)

        active_residues = int(residue_mask.sum(1).max())
        active_atom_mask = atom_mask[:, :active_residues]
        active_condition = residue_condition[:, :active_residues]
        atoms = self.atom_input(
            coordinates[:, :active_residues],
            None if self_conditioned_coordinates is None else self_conditioned_coordinates[:, :active_residues],
            residue_x[:, :active_residues], active_condition, active_atom_mask,
        )
        segment_index = layout.segment_index[:, :active_residues]
        for block in self.atom_encoder:
            atoms = checkpoint(
                block, atoms, active_condition, active_atom_mask, segment_index,
                enabled=self.config.checkpoint_blocks,
            )
        atom_skip = atoms
        atom_update = self.atom_to_residue(atoms, active_condition, active_atom_mask)
        atom_update = F.pad(atom_update, (0, 0, 0, residue_count - active_residues))
        residue_x = residue_x + atom_update

        tokens = torch.cat((registers, residue_x), 1)
        tokens = self._residue_stage(
            self.residue_encoder, tokens, token_condition, token_mask, token_positions
        )
        residue_skip = tokens[:, REGISTER_COUNT:]
        patches, patch_condition = self.patchify(residue_skip, residue_condition, layout)
        coarse_x = torch.cat((tokens[:, :REGISTER_COUNT], patches), 1)
        coarse_condition = torch.cat((register_condition, patch_condition), 1)
        coarse_mask = torch.cat((token_mask[:, :REGISTER_COUNT], layout.patch_mask), 1)
        patch_positions = layout.pack(token_positions[:, REGISTER_COUNT:])[:, :, 0]
        coarse_positions = torch.cat((register_positions, patch_positions), 1)

        ca_slots = layout.pack(raw_coordinates[..., 1, :])
        sc_ca_slots = None
        if raw_self_conditioned_coordinates is not None and self.config.pair_self_conditioned_geometry:
            sc_ca_slots = layout.pack(raw_self_conditioned_coordinates[..., 1, :])
        pair_dtype = torch.bfloat16 if raw_coordinates.is_cuda else coarse_x.dtype
        pair = self.pair_initializer(ca_slots, layout, pair_dtype, sc_ca_slots)
        structure_condition = (
            (residue_condition * residue_mask[..., None]).sum(1)
            / residue_mask.sum(1, keepdim=True).clamp_min(1).to(residue_condition.dtype)
        )
        intermediate_distograms = []
        for layer_index, block in enumerate(self.coarse):
            if self.config.checkpoint_blocks and torch.is_grad_enabled():
                coarse_x, pair = torch.utils.checkpoint.checkpoint(
                    block, coarse_x, coarse_condition, pair, coarse_mask,
                    coarse_positions, layout.pair_mask, use_reentrant=False,
                )
            else:
                coarse_x, pair = block(
                    coarse_x, coarse_condition, pair, coarse_mask, coarse_positions, layout.pair_mask
                )
            if compute_intermediate_distograms and layer_index + 1 < len(self.coarse):
                if self.intermediate_distogram is None:
                    raise ValueError(
                        "intermediate distograms requested but model.intermediate_distograms=false"
                    )
                intermediate = self.intermediate_distogram(pair, layout, structure_condition)
                intermediate_distograms.append(intermediate)
                if self.distogram_pair_feedback is not None:
                    pair = self.distogram_pair_feedback(
                        pair, intermediate.coarse_logits, layout.pair_mask
                    )

        residue_x = self.unpatchify(residue_skip, coarse_x[:, REGISTER_COUNT:], layout)
        tokens = torch.cat((coarse_x[:, :REGISTER_COUNT], residue_x), 1)
        tokens = self._residue_stage(
            self.residue_decoder, tokens, token_condition, token_mask, token_positions
        )
        residue_x = tokens[:, REGISTER_COUNT:]

        atoms = self.atom_output.inject(atom_skip, residue_x[:, :active_residues], active_atom_mask)
        for block in self.atom_decoder:
            atoms = checkpoint(
                block, atoms, active_condition, active_atom_mask, segment_index,
                enabled=self.config.checkpoint_blocks,
            )
        raw_update = self.atom_output.decode(atoms, active_atom_mask)
        raw_update = F.pad(raw_update, (0, 0, 0, 0, 0, residue_count - active_residues))
        aatype_logits = self.aatype_output(atoms, active_atom_mask, residue_count)
        secondary_structure_logits = (
            None
            if self.secondary_structure_output is None
            else self.secondary_structure_output(residue_x, residue_mask)
        )

        sigma_data_sq = self.config.sigma_data**2
        denominator = sigma.square() + sigma_data_sq
        c_skip = sigma_data_sq / denominator
        c_out = sigma * self.config.sigma_data / denominator.sqrt()
        predicted_coordinates = c_skip[..., None] * raw_coordinates + c_out[..., None] * raw_update
        predicted_coordinates = predicted_coordinates * atom_mask[..., None].to(predicted_coordinates.dtype)

        return Prediction(
            coordinates=predicted_coordinates,
            aatype_logits=aatype_logits,
            distogram=self.distogram(pair, layout) if compute_distogram else None,
            intermediate_distograms=tuple(intermediate_distograms),
            secondary_structure_logits=secondary_structure_logits,
        )

    def denoise(
        self,
        inputs: DenoiserInput,
        *,
        compute_distogram: bool = True,
        compute_intermediate_distograms: bool = False,
    ) -> Prediction:
        """Named alias used by the sampling runtime."""

        return self(
            inputs,
            compute_distogram=compute_distogram,
            compute_intermediate_distograms=compute_intermediate_distograms,
        )

    @torch.no_grad()
    def self_condition(self, inputs: DenoiserInput) -> Prediction:
        """Cheap first pass for stochastic self-conditioning."""

        return self(
            inputs,
            compute_distogram=False,
            compute_intermediate_distograms=False,
        )
