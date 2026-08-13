"""Strict checkpoints and dependency-free Atom14 PDB/FASTA output."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from torch import Tensor, nn

from koochak.storage import checkpoint as checkpoint_lib

from .config import RunConfig


RESTYPES = "ARNDCQEGHILKMFPSTWYV"
RESTYPE_3 = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)
ATOM14_NAMES = (
    ("N", "CA", "C", "O", "CB"),
    ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"),
    ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"),
    ("N", "CA", "C", "O", "CB", "SG"),
    ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"),
    ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"),
    ("N", "CA", "C", "O"),
    ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"),
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"),
    ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"),
    ("N", "CA", "C", "O", "CB", "CG", "SD", "CE"),
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    ("N", "CA", "C", "O", "CB", "CG", "CD"),
    ("N", "CA", "C", "O", "CB", "OG"),
    ("N", "CA", "C", "O", "CB", "OG1", "CG2"),
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    ("N", "CA", "C", "O", "CB", "CG1", "CG2"),
)
CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def load_checkpoint(
    model: nn.Module,
    checkpoint_file: str | Path,
    *,
    config: RunConfig | None = None,
    use_ema: bool = True,
) -> dict[str, Any]:
    """Strictly load raw weights, optionally replacing parameters with EMA values."""

    checkpoint = checkpoint_lib.load(str(checkpoint_file))
    if config is not None and checkpoint.get("config") is not None:
        saved_model = checkpoint["config"].get("model")
        if saved_model != asdict(config.model):
            raise ValueError("checkpoint model configuration does not match --config")
    raw = checkpoint.get("model")
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint has no model state dictionary")
    state = dict(checkpoint_lib.strip_module_prefix(dict(raw)))
    if use_ema:
        ema = checkpoint.get("ema")
        if not isinstance(ema, Mapping) or not isinstance(ema.get("shadow"), Mapping):
            raise ValueError("checkpoint has no EMA state; pass --raw to use training weights")
        shadow = checkpoint_lib.strip_module_prefix(dict(ema["shadow"]))
        unknown = set(shadow) - set(state)
        if unknown:
            raise ValueError(f"EMA contains unknown model keys: {sorted(unknown)[:5]}")
        state.update(shadow)
    model.load_state_dict(state, strict=True)
    return checkpoint


def sequence_string(aatype: Tensor, chain_lengths: Sequence[int]) -> str:
    """Convert one flat numeric sequence to slash-delimited chains."""

    values = aatype.detach().cpu().tolist()
    chains: list[str] = []
    start = 0
    for length in chain_lengths:
        chains.append("".join(RESTYPES[value] if 0 <= value < 20 else "X" for value in values[start:start + length]))
        start += length
    if start != len(values):
        raise ValueError("chain lengths do not sum to sequence length")
    return "/".join(chains)


def write_fasta(file: str | Path, aatype: Tensor, chain_lengths: Sequence[int], name: str) -> None:
    sequence = sequence_string(aatype, chain_lengths)
    Path(file).write_text(f">{name}\n{sequence}\n", encoding="utf-8")


def write_pdb(
    file: str | Path,
    coordinates: Tensor,
    aatype: Tensor,
    chain_lengths: Sequence[int],
) -> None:
    """Write one predicted Atom14 structure, omitting nonexistent sidechain slots."""

    if coordinates.ndim != 3 or tuple(coordinates.shape[-2:]) != (14, 3):
        raise ValueError("coordinates must have shape [N,14,3]")
    if coordinates.shape[0] != aatype.numel() or sum(chain_lengths) != aatype.numel():
        raise ValueError("coordinate, sequence, and chain lengths disagree")
    if len(chain_lengths) > len(CHAIN_IDS):
        raise ValueError(f"PDB output supports at most {len(CHAIN_IDS)} chains")
    xyz = coordinates.detach().float().cpu()
    sequence = aatype.detach().long().cpu()
    lines: list[str] = []
    serial = 1
    residue_offset = 0
    for chain_number, length in enumerate(chain_lengths):
        chain_id = CHAIN_IDS[chain_number]
        for residue_number in range(1, length + 1):
            index = residue_offset + residue_number - 1
            residue_type = int(sequence[index])
            residue_type = residue_type if 0 <= residue_type < 20 else 7
            residue_name = RESTYPE_3[residue_type]
            for atom_slot, atom_name in enumerate(ATOM14_NAMES[residue_type]):
                x, y, z = (float(value) for value in xyz[index, atom_slot])
                element = atom_name[0]
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:^4s} {residue_name:>3s} {chain_id}{residue_number:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2s}  "
                )
                serial += 1
        lines.append(f"TER   {serial:5d}      {residue_name:>3s} {chain_id}{length:4d}")
        serial += 1
        residue_offset += length
    lines.extend(("END", ""))
    Path(file).write_text("\n".join(lines), encoding="ascii")


def write_sample_batch(
    output_dir: str | Path,
    coordinates: Tensor,
    aatype: Tensor,
    chain_lengths: Sequence[int],
    *,
    start_index: int = 0,
) -> None:
    """Write paired PDB and FASTA files for a sampled batch."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for batch_index in range(coordinates.shape[0]):
        name = f"sample_{start_index + batch_index:05d}"
        write_pdb(output_dir / f"{name}.pdb", coordinates[batch_index], aatype[batch_index], chain_lengths)
        write_fasta(output_dir / f"{name}.fasta", aatype[batch_index], chain_lengths, name)


__all__ = [
    "load_checkpoint",
    "sequence_string",
    "write_fasta",
    "write_pdb",
    "write_sample_batch",
]
