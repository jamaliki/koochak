import torch

from hierarchical_kaveh.types import DenoiserInput, Prediction


def test_denoiser_input_moves_and_attaches_detached_self_conditioning() -> None:
    batch, residues = 2, 5
    atom_mask = torch.ones(batch, residues, 14, dtype=torch.bool)
    model_input = DenoiserInput(
        coordinates=torch.randn(batch, residues, 14, 3),
        sigma=torch.ones(batch, residues, 14),
        residue_index=torch.arange(residues).expand(batch, -1),
        chain_index=torch.zeros(batch, residues, dtype=torch.long),
        chain_break=torch.zeros(batch, residues),
        atom_mask=atom_mask,
    )
    prediction = Prediction(
        coordinates=torch.randn(batch, residues, 14, 3, requires_grad=True),
        aatype_logits=torch.randn(batch, residues, 20, requires_grad=True),
    )
    conditioned = model_input.with_self_conditioning(prediction)
    assert conditioned.self_conditioned_coordinates is not None
    assert conditioned.self_conditioning_mask.all()
    assert not conditioned.self_conditioned_coordinates.requires_grad
    assert conditioned.to("cpu").residue_mask.all()

    unconditioned = model_input.with_self_conditioning(None)
    assert torch.count_nonzero(unconditioned.self_conditioned_coordinates) == 0
    assert not unconditioned.self_conditioning_mask.any()
