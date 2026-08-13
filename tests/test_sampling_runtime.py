from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from hierarchical_kaveh.config import SamplingConfig
from hierarchical_kaveh.config import RunConfig
from hierarchical_kaveh.io import load_checkpoint, sequence_string, write_sample_batch
from hierarchical_kaveh.sampling import build_topology, parse_chain_lengths, sample, sigma_schedule
from hierarchical_kaveh.types import Prediction


class RecordingDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.inputs = []

    def forward(self, inputs, **kwargs):
        self.inputs.append((inputs, kwargs))
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


def test_sampler_uses_two_pass_coordinate_self_conditioning() -> None:
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
    assert torch.all(first.sigma == first.sigma[..., :1])
    assert first.self_conditioned_coordinates is None
    assert second.self_conditioned_coordinates is not None
    assert first_kwargs == second_kwargs == {"compute_distogram": False}
    assert len(model.inputs) == 2 * config.num_steps
    assert all(model.inputs[index][0].self_conditioned_coordinates is None for index in range(0, 6, 2))
    assert all(model.inputs[index][0].self_conditioned_coordinates is not None for index in range(1, 6, 2))
    assert result.coordinates.shape == (1, 4, 14, 3)


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
