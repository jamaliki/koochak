"""Canonical residue and Atom14 topology shared by data and output code."""

from __future__ import annotations

import torch
from torch import Tensor


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
    ("N", "CA", "C", "O", "CB", "CG", "SG"),
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
ATOM14_COUNTS = torch.tensor([len(names) for names in ATOM14_NAMES], dtype=torch.long)


def physical_atom14_mask(aatype: Tensor) -> Tensor:
    """Return slots occupied by real atoms for canonical residue types."""

    safe = aatype.clamp(0, len(ATOM14_NAMES) - 1)
    counts = ATOM14_COUNTS.to(device=aatype.device)[safe]
    slots = torch.arange(14, device=aatype.device)
    valid_residue = aatype.ge(0) & aatype.lt(len(ATOM14_NAMES))
    return (slots < counts[..., None]) & valid_residue[..., None]


__all__ = ["ATOM14_NAMES", "RESTYPE_3", "RESTYPES", "physical_atom14_mask"]
