"""Minimal single-graph Progres v1.1.0 inference without PyG or torch-scatter."""

from __future__ import annotations

from math import ceil
from pathlib import Path

import torch
from torch import nn
from torch.nn.functional import normalize

EMBEDDING_SIZE = 128
CONTACT_DISTANCE = 10.0


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        channels = int(ceil(channels / 2) * 2)
        frequencies = 1.0 / (2000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", frequencies)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        angles = torch.einsum("...i,j->...ij", tensor, self.inv_freq)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


def _batched_index_select(values: torch.Tensor, indices: torch.Tensor, dim: int = 1) -> torch.Tensor:
    value_dims = values.shape[(dim + 1):]
    indices_shape = list(indices.shape)
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * expand_len), ...)]
    expand_shape = [-1] * len(values.shape)
    expand_shape[slice(dim, dim + expand_len)] = indices.shape[slice(dim, dim + expand_len)]
    values = values.expand(*expand_shape)
    return values.gather(dim + expand_len, indices)


class EGNN(nn.Module):
    def __init__(self, dim: int = 128, message_dim: int = 64) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear((dim * 2) + 1, 256), nn.Identity(), nn.SiLU(),
            nn.Linear(256, message_dim), nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(dim + message_dim, dim * 2), nn.Identity(), nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(
        self, features: torch.Tensor, coordinates: torch.Tensor,
        mask: torch.Tensor, adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, residue_count, _ = features.shape
        relative = coordinates[:, :, None, :] - coordinates[:, None, :, :]
        distances = relative.square().sum(dim=-1, keepdim=True)
        ranking = distances[..., 0].clone()
        ranking.masked_fill_(~(mask[:, :, None] * mask[:, None, :]), 1e5)
        neighbor_count = int(adjacency.float().sum(dim=-1).max().item())
        self_mask = torch.eye(residue_count, device=features.device, dtype=torch.bool)[None]
        adjacency = adjacency.masked_fill(self_mask, False)
        ranking.masked_fill_(self_mask, -1.0)
        ranking.masked_fill_(adjacency, 0.0)
        neighbor_ranks, neighbor_indices = ranking.topk(neighbor_count, dim=-1, largest=False)
        neighbor_mask = neighbor_ranks <= 0
        relative = _batched_index_select(relative, neighbor_indices, dim=2)
        distances = _batched_index_select(distances, neighbor_indices, dim=2)
        neighbor_features = _batched_index_select(features, neighbor_indices, dim=1)
        source_features = features[:, :, None, :]
        source_features, neighbor_features = torch.broadcast_tensors(source_features, neighbor_features)
        messages = self.edge_mlp(torch.cat((source_features, neighbor_features, distances), dim=-1))
        valid = mask[:, :, None] * _batched_index_select(mask, neighbor_indices, dim=1) * neighbor_mask
        messages = messages.masked_fill(~valid[..., None], 0.0)
        output = self.node_mlp(torch.cat((features, messages.sum(dim=-2)), dim=-1)) + features
        return output, coordinates


class ProgresModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.node_enc = nn.Linear(68, 128)
        self.layers = nn.ModuleList([EGNN() for _ in range(6)])
        self.node_dec = nn.Sequential(
            nn.Linear(128, 128), nn.Identity(), nn.SiLU(), nn.Linear(128, 128),
        )
        self.graph_dec = nn.Sequential(
            nn.Linear(128, 128), nn.Identity(), nn.SiLU(), nn.Identity(),
            nn.Linear(128, EMBEDDING_SIZE),
        )

    def forward(self, features: torch.Tensor, coordinates: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        features, coordinates = features[None], coordinates[None]
        mask = torch.ones(1, features.shape[1], dtype=torch.bool, device=features.device)
        features = self.node_enc(features)
        for layer in self.layers:
            features, coordinates = layer(features, coordinates, mask, adjacency[None])
        graph_features = self.node_dec(features).squeeze(0).sum(dim=0, keepdim=True)
        return normalize(self.graph_dec(graph_features), dim=1).squeeze(0)


def read_ca_coordinates(file: Path) -> torch.Tensor:
    coordinates = []
    chain_id = None
    for line in file.read_text(encoding="utf-8").splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
            continue
        if chain_id is None:
            chain_id = line[21]
        elif line[21] != chain_id:
            break
        coordinates.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if len(coordinates) < 4:
        raise ValueError(f"expected at least four CA coordinates in {file}")
    return torch.tensor(coordinates)


def featurize(coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    residue_count = len(coordinates)
    distances = torch.cdist(coordinates[None], coordinates[None], compute_mode="donot_use_mm_for_euclid_dist")
    adjacency = (distances <= CONTACT_DISTANCE).squeeze(0)
    normalized_degrees = (adjacency.sum(dim=0) / adjacency.sum(dim=0).max()).unsqueeze(1)
    termini = torch.zeros(residue_count, 2)
    termini[0, 0] = termini[-1, 1] = 1.0
    ab = coordinates[1:-2] - coordinates[:-3]
    bc = coordinates[2:-1] - coordinates[1:-2]
    cd = coordinates[3:] - coordinates[2:-1]
    cross_ab_bc = torch.cross(ab, bc, dim=1)
    cross_bc_cd = torch.cross(bc, cd, dim=1)
    torsions = torch.atan2(
        (torch.cross(cross_ab_bc, cross_bc_cd, dim=1) * normalize(bc, dim=1)).sum(dim=1),
        (cross_ab_bc * cross_bc_cd).sum(dim=1),
    )
    torsions = torch.cat((torch.zeros(1), torsions / torch.pi, torch.zeros(2))).unsqueeze(1)
    positions = SinusoidalPositionalEncoding(64)(torch.arange(1, residue_count + 1))
    return torch.cat((normalized_degrees, termini, torsions, positions), dim=1), adjacency


def load_model(weights_file: Path) -> ProgresModel:
    checkpoint = torch.load(weights_file, map_location="cpu", weights_only=True)
    model = ProgresModel()
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def embed_structure(model: ProgresModel, file: Path) -> torch.Tensor:
    coordinates = read_ca_coordinates(file)
    features, adjacency = featurize(coordinates)
    with torch.no_grad():
        return model(features, coordinates, adjacency)
