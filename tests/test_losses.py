import torch

from hierarchical_kaveh.config import LossConfig, ModelConfig
from hierarchical_kaveh.diffusion import (
    align_coordinates_to_reference,
    aligned_edm_loss,
    aatype_cross_entropy,
    aatype_marginal_js,
    aatype_sigma_weights,
    compute_losses,
    distogram_cross_entropy,
    edm_coordinate_loss,
    secondary_structure_cross_entropy,
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


def test_unaligned_edm_coordinate_loss_penalizes_rigid_motion() -> None:
    torch.manual_seed(2)
    target = torch.randn(1, 5, 14, 3)
    mask = torch.ones(1, 5, 14, dtype=torch.bool)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    prediction = target @ rotation + torch.tensor([4.0, -3.0, 2.0])
    loss = edm_coordinate_loss(
        prediction, target, mask, torch.ones(1), align_target=False
    )
    assert loss > 1.0


def test_coordinate_frame_alignment_recovers_rigid_motion_and_masks_padding() -> None:
    torch.manual_seed(4)
    reference = torch.randn(2, 5, 14, 3)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    source = reference @ rotation + torch.tensor([4.0, -3.0, 2.0])
    mask = torch.ones(2, 5, 14, dtype=torch.bool)
    mask[:, -1, 7:] = False
    aligned = align_coordinates_to_reference(source, reference, mask)
    torch.testing.assert_close(aligned[mask], reference[mask], atol=2e-5, rtol=0.0)
    assert torch.equal(aligned[~mask], torch.zeros_like(aligned[~mask]))


def test_coordinate_frame_alignment_uses_translation_only_for_two_points() -> None:
    reference = torch.tensor([[[[0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]]])
    source = reference + torch.tensor([[[[3.0, -2.0, 1.0]], [[3.0, -2.0, 1.0]]]])
    mask = torch.ones(1, 2, 1, dtype=torch.bool)
    aligned = align_coordinates_to_reference(source, reference, mask)
    torch.testing.assert_close(aligned, reference)


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


def test_aatype_loss_uses_only_selected_samples() -> None:
    logits = torch.zeros(2, 2, 20, requires_grad=True)
    target = torch.tensor([[0, 1], [2, 3]])
    mask = torch.ones(2, 2, dtype=torch.bool)
    loss = aatype_cross_entropy(
        logits,
        target,
        mask,
        sample_mask=torch.tensor([True, False]),
        polar_aatypes="R",
        polar_weight=2.0,
    )
    expected = torch.nn.functional.cross_entropy(logits[0].float(), target[0], reduction="none")
    torch.testing.assert_close(loss, (expected[0] + 2 * expected[1]) / 3)
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[0]) > 0
    assert torch.count_nonzero(logits.grad[1]) == 0


def test_aatype_loss_all_inactive_is_differentiable_zero() -> None:
    logits = torch.randn(2, 3, 20, requires_grad=True)
    loss = aatype_cross_entropy(
        logits,
        torch.zeros(2, 3, dtype=torch.long),
        torch.ones(2, 3, dtype=torch.bool),
        sample_mask=torch.zeros(2, dtype=torch.bool),
    )
    torch.testing.assert_close(loss, torch.zeros(()))
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0


def test_aatype_sigma_weights_support_hard_and_linear_gates() -> None:
    sigma = torch.tensor([0.25, 0.5, 0.75, 1.0, 2.0])
    torch.testing.assert_close(
        aatype_sigma_weights(sigma, full_max=0.5),
        torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        aatype_sigma_weights(sigma, full_max=0.5, ramp_max=1.0),
        torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        aatype_sigma_weights(sigma, full_max=0.5, ramp_max=2.0),
        torch.tensor([1.0, 1.0, 5.0 / 6.0, 2.0 / 3.0, 0.0]),
    )


def test_aatype_fractional_weights_scale_gradients_and_keep_zero_samples_zero() -> None:
    logits = torch.zeros(2, 1, 20, requires_grad=True)
    target = torch.tensor([[0], [1]])
    mask = torch.ones(2, 1, dtype=torch.bool)
    loss = aatype_cross_entropy(
        logits,
        target,
        mask,
        sample_weights=torch.tensor([0.25, 0.0]),
        polar_aatypes="",
        polar_weight=1.0,
    )
    loss.backward()
    assert torch.count_nonzero(logits.grad[0]) > 0
    assert torch.count_nonzero(logits.grad[1]) == 0


def test_aatype_marginal_js_is_mask_correct_and_differentiable() -> None:
    logits = torch.full((2, 2, 20), -20.0, requires_grad=True)
    logits.data[..., :2] = 0.0
    target = torch.tensor([[0, 1], [0, 1]])
    mask = torch.tensor([[True, True], [True, False]])
    matched = aatype_marginal_js(
        logits, target, mask, sample_weights=torch.tensor([1.0, 0.0])
    )
    assert matched < 1.0e-5
    matched.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[1]) == 0

    collapsed = torch.full((2, 2, 20), -20.0)
    collapsed[..., 0] = 20.0
    assert aatype_marginal_js(
        collapsed, target, mask, sample_weights=torch.tensor([1.0, 0.0])
    ) > matched.detach()


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
        "model_atom_mask": atom_mask,
        "coordinate_mask": atom_mask,
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
        "aatype_active_fraction",
        "aatype_marginal_js_loss",
        "aatype_sigma_weight_mean",
        "smooth_lddt_loss",
        "distogram_loss",
        "secondary_structure_loss",
    }
    torch.testing.assert_close(losses["coordinate_loss"], torch.zeros(()))


def test_compute_losses_gates_sequence_ce_at_inclusive_sigma_boundary() -> None:
    batch_size, residues = 3, 2
    atom_mask = torch.ones(batch_size, residues, 14, dtype=torch.bool)
    target = torch.randn(batch_size, residues, 14, 3)
    sigma = torch.tensor([0.49, 0.5, 1.0])
    logits = torch.zeros(batch_size, residues, 20, requires_grad=True)
    inputs = DenoiserInput(
        coordinates=target.clone(),
        sigma=sigma[:, None, None].expand(-1, residues, 14),
        residue_index=torch.arange(residues)[None].expand(batch_size, -1),
        chain_index=torch.zeros(batch_size, residues, dtype=torch.long),
        chain_break=torch.zeros(batch_size, residues, dtype=torch.bool),
        atom_mask=atom_mask,
        aatype_input=torch.full((batch_size, residues), 20),
    )
    batch = {
        "x0": target,
        "sigma": sigma,
        "model_atom_mask": atom_mask,
        "coordinate_mask": atom_mask,
        "residue_mask": atom_mask[..., 1],
        "aatype": torch.zeros(batch_size, residues, dtype=torch.long),
    }
    losses = compute_losses(
        Prediction(coordinates=target, aatype_logits=logits),
        inputs,
        batch,
        LossConfig(
            coordinate_weight=0.0,
            aatype_weight=1.0,
            aatype_sigma_max=0.5,
            smooth_lddt_weight=0.0,
            distogram_weight=0.0,
            polar_weight=1.0,
        ),
        ModelConfig(),
    )
    torch.testing.assert_close(losses["aatype_active_fraction"], torch.tensor(2 / 3))
    losses["loss"].backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:2]) > 0
    assert torch.count_nonzero(logits.grad[2]) == 0


def test_secondary_structure_cross_entropy_ignores_unknown_and_padding() -> None:
    logits = torch.zeros(1, 4, 3, requires_grad=True)
    target = torch.tensor([[0, 1, 2, 3]])
    mask = torch.tensor([[True, True, True, False]])
    loss = secondary_structure_cross_entropy(logits, target, mask)
    torch.testing.assert_close(loss, torch.log(torch.tensor(3.0)))
    loss.backward()
    assert torch.count_nonzero(logits.grad[0, 3]) == 0


def test_compute_losses_includes_weighted_secondary_structure_objective() -> None:
    batch_size, residues = 1, 3
    atom_mask = torch.ones(batch_size, residues, 14, dtype=torch.bool)
    target = torch.randn(batch_size, residues, 14, 3)
    ss_logits = torch.zeros(batch_size, residues, 3, requires_grad=True)
    batch = {
        "x0": target,
        "sigma": torch.full((batch_size,), 0.25),
        "coordinate_mask": atom_mask,
        "residue_mask": atom_mask[..., 1],
        "aatype": torch.zeros(batch_size, residues, dtype=torch.long),
        "secondary_structure": torch.tensor([[0, 1, 2]]),
    }
    losses = compute_losses(
        Prediction(
            coordinates=target,
            aatype_logits=torch.zeros(batch_size, residues, 20),
            secondary_structure_logits=ss_logits,
        ),
        DenoiserInput(
            coordinates=target,
            sigma=torch.full((batch_size, residues, 14), 0.25),
            residue_index=torch.arange(residues)[None],
            chain_index=torch.zeros(batch_size, residues, dtype=torch.long),
            chain_break=torch.zeros(batch_size, residues, dtype=torch.bool),
            atom_mask=atom_mask,
        ),
        batch,
        LossConfig(
            coordinate_weight=0.0,
            aatype_weight=0.0,
            smooth_lddt_weight=0.0,
            distogram_weight=0.0,
            secondary_structure_weight=1.0,
        ),
        ModelConfig(),
    )
    torch.testing.assert_close(losses["loss"], losses["secondary_structure_loss"])
    losses["loss"].backward()
    assert ss_logits.grad is not None
