from dataclasses import replace
from unittest.mock import Mock

import torch
from torch import nn

from hierarchical_kaveh.config import LossConfig, ModelConfig
from hierarchical_kaveh.model import HierarchicalKaveh
from hierarchical_kaveh.training import PallatomTrainingStep
from hierarchical_kaveh.types import Prediction
from hierarchical_kaveh.types import DenoiserInput
from scripts.prepare_progres_condition_bank import _select_diverse
from scripts.sample_progres_conditioned_milestone import paired_seed


def _config(**changes):
    values = dict(node_dim=32, condition_dim=16, pair_dim=8, atom_dim=16,
                  attention_heads=4, attention_head_dim=8, atom_heads=2,
                  atom_head_dim=8, atom_encoder_depth=1, residue_encoder_depth=1,
                  coarse_depth=1, residue_decoder_depth=1, atom_decoder_depth=1,
                  residue_ffn_expansion=2, atom_ffn_expansion=2, dropout=0.0,
                  pair_rbf_bins=4, distogram_bins=8, progres_conditioning=True)
    values.update(changes)
    return ModelConfig(**values)


def _input(batch=2, length=8):
    mask = torch.ones(batch, length, 14, dtype=torch.bool)
    return DenoiserInput(
        coordinates=torch.randn(batch, length, 14, 3), sigma=torch.ones(batch, length, 14),
        residue_index=torch.arange(length).expand(batch, -1),
        chain_index=torch.zeros(batch, length, dtype=torch.long),
        chain_break=torch.zeros(batch, length), atom_mask=mask,
    )


def test_null_condition_is_exactly_equivalent_at_zero_projection():
    model = HierarchicalKaveh(_config()).eval()
    inputs = _input()
    null = replace(inputs, progres_embedding=torch.randn(2, 128), progres_conditioning_mask=torch.zeros(2, dtype=torch.bool))
    with torch.no_grad():
        left = model(inputs, compute_distogram=False)
        right = model(null, compute_distogram=False)
    torch.testing.assert_close(left.coordinates, right.coordinates)
    torch.testing.assert_close(left.aatype_logits, right.aatype_logits)


def test_condition_changes_output_after_final_projection_is_nonzero():
    model = HierarchicalKaveh(_config()).eval()
    with torch.no_grad():
        model.progres_condition_projection[-1].weight.normal_(0.0, 0.01)
        for module in model.modules():
            if hasattr(module, "modulation"):
                module.modulation.weight.normal_(0.0, 0.01)
        model.atom_output.output.weight.fill_(0.01)
    inputs = _input()
    embedding = torch.randn(2, 128)
    conditioned = replace(inputs, progres_embedding=embedding, progres_conditioning_mask=torch.ones(2, dtype=torch.bool))
    residue_mask = inputs.residue_mask
    sigma = inputs.sigma
    with torch.no_grad():
        _sigma, left, _left_global = model._conditions(sigma, residue_mask, inputs)
        _sigma, right, _right_global = model._conditions(sigma, residue_mask, conditioned)
    assert not torch.equal(left, right)


def test_condition_bank_selection_and_paired_seed_are_deterministic():
    vectors = torch.eye(8, 128).numpy().astype("float32")
    ids = [f"id-{index}" for index in range(8)]
    assert _select_diverse(ids, vectors, 4, 11) == _select_diverse(ids, vectors, 4, 11)
    assert paired_seed(7, 3) == 7 + 1_000_003 * 3


def test_zero_initialized_final_projection_and_shared_dropout_mask():
    model = HierarchicalKaveh(_config()).eval()
    assert torch.count_nonzero(model.progres_condition_projection[-1].weight) == 0
    step = PallatomTrainingStep(LossConfig(), _config(), self_conditioning_probability=1.0, seed=9)
    batch = {
        "progres_embedding": torch.randn(2, 128),
        "x_t": torch.randn(2, 8, 14, 3), "t": torch.ones(2, 8, 14),
        "model_atom_mask": torch.ones(2, 8, 14, dtype=torch.bool),
        "coordinate_mask": torch.ones(2, 8, 14, dtype=torch.bool),
        "residue_mask": torch.ones(2, 8, dtype=torch.bool),
        "aatype_input": torch.full((2, 8), 20), "res_idx": torch.arange(8).expand(2, -1),
        "chain_idx": torch.zeros(2, 8, dtype=torch.long),
        "chain_breaks_per_residue": torch.zeros(2, 8, dtype=torch.bool),
        "patch_capacity": 4,
    }
    prediction = Prediction(batch["x_t"], torch.zeros(2, 8, 20))

    class RecordingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, inputs, **_kwargs):
            self.inputs.append(inputs)
            return prediction

    model_recorder = RecordingModel()
    zero = prediction.coordinates.sum() * 0 + 1.0
    losses = {name: zero for name in ("loss", "coordinate_loss", "aatype_loss", "aatype_active_fraction", "smooth_lddt_loss", "distogram_loss")}
    import hierarchical_kaveh.training as training_module
    original = training_module.compute_losses
    training_module.compute_losses = Mock(return_value=losses)
    try:
        step(model_recorder, batch, {"autocast": torch.no_grad})
    finally:
        training_module.compute_losses = original
    assert len(model_recorder.inputs) == 2
    assert torch.equal(model_recorder.inputs[0].progres_conditioning_mask, model_recorder.inputs[1].progres_conditioning_mask)
