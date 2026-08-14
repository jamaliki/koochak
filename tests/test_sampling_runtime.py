from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

import hierarchical_kaveh.sampling as sampling_module
from hierarchical_kaveh.config import SamplingConfig
from hierarchical_kaveh.config import RunConfig
from hierarchical_kaveh.diffusion.corruption import random_rigid_augmentation
from hierarchical_kaveh.io import load_checkpoint, sequence_string, write_sample_batch
from hierarchical_kaveh.sampling import (
    _augment_batch,
    _churn_gamma,
    build_topology,
    parse_chain_lengths,
    sample,
    sigma_schedule,
)
from hierarchical_kaveh.types import Prediction


class RecordingDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.inputs = []

    def forward(self, inputs, **kwargs):
        self.inputs.append((inputs, kwargs))
        batch, residues = inputs.coordinates.shape[:2]
        call = len(self.inputs)
        logits = torch.zeros(batch, residues, 20, device=inputs.coordinates.device)
        logits[..., call % 20] = 1.0
        return Prediction(
            coordinates=torch.full_like(inputs.coordinates, float(call)),
            aatype_logits=logits,
            distogram=None,
        )


class ZeroDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, inputs, **kwargs):
        self.inputs.append(inputs)
        batch, residues = inputs.coordinates.shape[:2]
        return Prediction(
            coordinates=torch.zeros_like(inputs.coordinates),
            aatype_logits=torch.zeros(batch, residues, 20, device=inputs.coordinates.device),
            distogram=None,
        )


def test_chain_topology_uses_index_gap_not_synthetic_break() -> None:
    topology = build_topology((3, 2), 2, "cpu")
    assert topology.residue_index[0].tolist() == [1, 2, 3, 67, 68]
    assert topology.chain_index[0].tolist() == [0, 0, 0, 1, 1]
    assert not topology.chain_break.any()


def test_chain_length_parser_and_schedule() -> None:
    assert parse_chain_lengths("96, 128") == (96, 128)
    config = replace(SamplingConfig(), num_steps=4)
    sigmas = sigma_schedule(config, "cpu")
    assert sigmas.shape == (4,)
    assert torch.all(sigmas[:-1] > sigmas[1:])


def test_sampler_carries_previous_prediction_as_self_conditioning() -> None:
    config = replace(SamplingConfig(), num_steps=3)
    model = RecordingDenoiser()
    result = sample(
        model,
        (4,),
        batch_size=1,
        config=config,
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(7),
    )
    first, first_kwargs = model.inputs[0]
    second, second_kwargs = model.inputs[1]
    third, third_kwargs = model.inputs[2]
    assert torch.all(first.sigma == first.sigma[..., :1])
    assert first.self_conditioned_coordinates is None
    assert second.self_conditioned_coordinates is not None
    assert third.self_conditioned_coordinates is not None
    assert torch.isfinite(second.self_conditioned_coordinates).all()
    assert torch.isfinite(third.self_conditioned_coordinates).all()
    assert first_kwargs == second_kwargs == third_kwargs == {"compute_distogram": False}
    assert len(model.inputs) == config.num_steps
    assert torch.all(result.aatype == 3)
    assert result.coordinates.shape == (1, 4, 14, 3)


def test_sampling_applies_the_same_rigid_frame_to_state_and_self_conditioning() -> None:
    coordinates = torch.randn(2, 4, 14, 3, generator=torch.Generator().manual_seed(3))
    self_conditioning = coordinates + torch.tensor([1.0, 2.0, 3.0])
    atom_mask = torch.ones(2, 4, 14, dtype=torch.bool)
    baseline_generator = torch.Generator().manual_seed(13)
    aligned_generator = torch.Generator().manual_seed(13)
    baseline = torch.stack(
        [
            random_rigid_augmentation(
                sample,
                mask,
                baseline_generator,
                translation_std=1.0,
            )
            for sample, mask in zip(coordinates, atom_mask, strict=True)
        ]
    )
    transformed, transformed_sc = _augment_batch(
        coordinates,
        self_conditioning,
        atom_mask,
        aligned_generator,
        1.0,
    )

    torch.testing.assert_close(transformed, baseline)
    assert transformed_sc is not None
    torch.testing.assert_close(
        (transformed_sc - transformed).norm(dim=-1),
        (self_conditioning - coordinates).norm(dim=-1),
    )
    assert torch.equal(
        torch.rand(16, generator=baseline_generator),
        torch.rand(16, generator=aligned_generator),
    )


def test_sampler_coordinates_match_declared_sigma_on_every_step(monkeypatch) -> None:
    monkeypatch.setattr(
        sampling_module,
        "_augment_batch",
        lambda coordinates, self_conditioning, atom_mask, generator, translation_std:
        (coordinates, self_conditioning),
    )
    model = ZeroDenoiser()
    config = SamplingConfig(
        num_steps=4,
        gamma=0.0,
        noise_scale=1.0,
        step_scale=1.0,
        translation_std=0.0,
    )
    sample(
        model,
        (4096,),
        batch_size=1,
        config=config,
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(31),
    )

    for inputs in model.inputs:
        observed_rms = inputs.coordinates.square().mean().sqrt()
        declared_sigma = inputs.sigma[0, 0, 0]
        torch.testing.assert_close(observed_rms, declared_sigma, rtol=0.03, atol=0.0)


def test_churn_uses_inclusive_perturbed_pallatom_time_gate() -> None:
    config = SamplingConfig(num_steps=200, gamma=0.2)
    at_min = torch.tensor(0.01, dtype=torch.float64)
    below_min = torch.tensor(0.009999, dtype=torch.float64)
    above_max = torch.tensor(1.000001, dtype=torch.float64)
    assert _churn_gamma(0, config, at_min).item() == config.gamma
    assert _churn_gamma(0, config, below_min).item() == 0.0
    assert _churn_gamma(0, config, above_max).item() == 0.0


def test_final_sequence_decode_is_deterministic_argmax() -> None:
    config = replace(SamplingConfig(), num_steps=1, sequence_temperature=0.1)
    first_model = RecordingDenoiser()
    second_model = RecordingDenoiser()
    first = sample(
        first_model,
        (4,),
        batch_size=1,
        config=config,
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(1),
    )
    second = sample(
        second_model,
        (4,),
        batch_size=1,
        config=config,
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(999),
    )
    assert torch.equal(first.aatype, second.aatype)
    assert torch.all(first.aatype == 1)


def test_output_writes_paired_pdb_and_fasta(tmp_path: Path) -> None:
    coordinates = torch.zeros(1, 3, 14, 3)
    aatype = torch.tensor([[0, 7, 19]])
    write_sample_batch(tmp_path, coordinates, aatype, (2, 1))
    fasta = (tmp_path / "sample_00000.fasta").read_text()
    pdb = (tmp_path / "sample_00000.pdb").read_text()
    assert sequence_string(aatype[0], (2, 1)) == "AG/V"
    assert fasta == ">sample_00000\nAG/V\n"
    assert " A   1" in pdb and " B   1" in pdb
    assert pdb.endswith("END\n")


def test_strict_koochak_checkpoint_loads_raw_and_ema(tmp_path: Path) -> None:
    config = RunConfig()
    source = nn.Linear(2, 1)
    with torch.no_grad():
        source.weight.fill_(3.0)
        source.bias.fill_(4.0)
    ema = {name: torch.full_like(value, 7.0) for name, value in source.state_dict().items()}
    checkpoint_file = tmp_path / "step0000001.pt"
    torch.save(
        {
            "model": source.state_dict(),
            "ema": {"shadow": ema},
            "config": config.to_dict(),
            "step": 1,
        },
        checkpoint_file,
    )

    target = nn.Linear(2, 1)
    loaded = load_checkpoint(target, checkpoint_file, config=config)
    assert loaded["step"] == 1
    assert all(torch.equal(value, torch.full_like(value, 7.0)) for value in target.state_dict().values())

    load_checkpoint(target, checkpoint_file, config=config, use_ema=False)
    assert torch.equal(target.weight, torch.full_like(target.weight, 3.0))
    assert torch.equal(target.bias, torch.full_like(target.bias, 4.0))


def test_checkpoint_overlays_partial_trainable_ema_on_strict_raw_state(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    checkpoint_file = tmp_path / "broken.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "ema": {"shadow": {"weight": torch.zeros_like(model.weight)}},
            "config": RunConfig().to_dict(),
        },
        checkpoint_file,
    )
    # A partial EMA is valid because Koochak tracks trainable parameters only;
    # missing entries retain their strict raw checkpoint values.
    load_checkpoint(model, checkpoint_file, config=RunConfig())
    assert torch.equal(model.weight, torch.zeros_like(model.weight))
    assert model.bias.shape == (1,)


def test_checkpoint_loader_normalizes_compile_and_ddp_wrappers(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    checkpoint_file = tmp_path / "compiled.pt"
    torch.save(
        {
            "model": {
                "module._orig_mod.weight": torch.full_like(model.weight, 3.0),
                "module._orig_mod.bias": torch.full_like(model.bias, 4.0),
            },
            "ema": {
                "shadow": {
                    "_orig_mod.module.weight": torch.full_like(model.weight, 7.0),
                    "_orig_mod.module.bias": torch.full_like(model.bias, 8.0),
                },
            },
            "config": RunConfig().to_dict(),
        },
        checkpoint_file,
    )
    load_checkpoint(model, checkpoint_file, config=RunConfig())
    assert torch.equal(model.weight, torch.full_like(model.weight, 7.0))
    assert torch.equal(model.bias, torch.full_like(model.bias, 8.0))
