import torch

from hierarchical_kaveh.config import LossConfig, ModelConfig
from hierarchical_kaveh.diffusion import (
    aligned_edm_loss,
    aatype_cross_entropy,
    compute_losses,
    distogram_cross_entropy,
    smooth_lddt_loss,
)
from hierarchical_kaveh.types import CompactDistogram, DenoiserInput, Prediction


def test_kabsch_aligned_edm_loss_ignores_rigid_motion() -> None:
    torch.manual_seed(2)
    target = torch.randn(2, 5, 14, 3)
    mask = torch.ones(2, 5, 14, dtype=torch.bool)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    prediction = target @ rotation + torch.tensor([4.0, -3.0, 2.0])
    loss = aligned_edm_loss(prediction, target, mask, torch.ones(2))
    torch.testing.assert_close(loss, torch.zeros(()), atol=2e-6, rtol=0.0)


def test_aatype_loss_applies_explicit_polar_weights() -> None:
    logits = torch.tensor([[[0.0] * 20, [0.0] * 20]])
    logits[0, 0, 0] = 3.0  # easy alanine
    target = torch.tensor([[0, 1]])  # A, R
    mask = torch.ones(1, 2, dtype=torch.bool)
    weighted = aatype_cross_entropy(
        logits,
        target,
        mask,
        polar_aatypes="R",
        polar_weight=2.0,
    )
    errors = torch.nn.functional.cross_entropy(
        logits.movedim(-1, 1), target, reduction="none"
    )
    torch.testing.assert_close(weighted, (errors[0, 0] + 2 * errors[0, 1]) / 3)


def test_smooth_lddt_is_rigid_invariant_and_penalizes_distortion() -> None:
    target = torch.randn(1, 3, 14, 3)
    mask = torch.ones(1, 3, 14, dtype=torch.bool)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rigid = target @ rotation + 7.0
    distorted = rigid.clone()
    distorted[:, 0, 0, 0] += 8.0
    rigid_loss = smooth_lddt_loss(rigid, target, mask, chunk_size=7)
    distorted_loss = smooth_lddt_loss(distorted, target, mask, chunk_size=7)
    assert distorted_loss > rigid_loss


def test_compact_patch_distogram_matches_dense_for_non_divisible_length() -> None:
    torch.manual_seed(3)
    batch, residues, patches, bins = 1, 5, 2, 8
    coarse = torch.randn(batch, patches, patches, bins, requires_grad=True)
    slot = torch.randn(4, 4, bins, requires_grad=True)
    residue_to_patch = torch.tensor([[0, 0, 0, 0, 1]])
    residue_slot = torch.tensor([[0, 1, 2, 3, 0]])
    patch_residue_index = torch.tensor([[[0, 1, 2, 3], [4, -1, -1, -1]]])
    residue_mask = torch.ones(batch, residues, dtype=torch.bool)
    compact = CompactDistogram(
        coarse,
        slot,
        residue_to_patch,
        residue_slot,
        patch_residue_index,
        residue_mask,
        True,
    )
    batch_index = torch.arange(batch)[:, None, None]
    dense = coarse[
        batch_index, residue_to_patch[:, :, None], residue_to_patch[:, None, :]
    ] + slot[residue_slot[:, :, None], residue_slot[:, None, :]]
    dense = dense + dense.transpose(1, 2)
    coordinates = torch.randn(batch, residues, 14, 3)
    compact_loss = distogram_cross_entropy(
        compact, coordinates, residue_mask, bins=bins, implementation="eager"
    )
    dense_loss = distogram_cross_entropy(dense, coordinates, residue_mask, bins=bins)
    torch.testing.assert_close(compact_loss, dense_loss)
    compact_loss.backward()
    assert coarse.grad is not None and slot.grad is not None


def test_compute_losses_has_only_supported_final_objectives() -> None:
    batch_size, residues = 1, 5
    atom_mask = torch.ones(batch_size, residues, 14, dtype=torch.bool)
    sigma = torch.full((batch_size, residues, 14), 0.25)
    target = torch.randn(batch_size, residues, 14, 3)
    inputs = DenoiserInput(
        coordinates=target.clone(),
        sigma=sigma,
        residue_index=torch.arange(residues)[None],
        chain_index=torch.zeros(batch_size, residues, dtype=torch.long),
        chain_break=torch.zeros(batch_size, residues, dtype=torch.bool),
        atom_mask=atom_mask,
        aatype_input=torch.full((batch_size, residues), 20),
    )
    batch = {
        "x0": target,
        "sigma": torch.full((batch_size,), 0.25),
        "atom14_mask": atom_mask,
        "residue_mask": atom_mask[..., 1],
        "aatype": torch.arange(residues)[None],
    }
    prediction = Prediction(
        coordinates=target,
        aatype_logits=torch.zeros(batch_size, residues, 20),
    )
    losses = compute_losses(
        prediction,
        inputs,
        batch,
        LossConfig(distogram_weight=0.0),
        ModelConfig(),
    )
    assert set(losses) == {
        "loss",
        "coordinate_loss",
        "aatype_loss",
        "smooth_lddt_loss",
        "distogram_loss",
    }
    torch.testing.assert_close(losses["coordinate_loss"], torch.zeros(()))
