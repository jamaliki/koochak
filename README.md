# Hierarchical Kaveh

Hierarchical Kaveh is the clean, single-model repository for Kaveh's p=4
hierarchical Atom14 diffusion architecture. It contains only what is needed to
train the model on existing ragged NPZ shards and sample unconditional protein
sequence/structure pairs.

## Information flow

```text
noisy Atom14 coordinates + amino-acid tokens
  -> residue-index-local atom attention
  -> learned atom pooling + explicit mean residual
  -> global residue attention
  -> chain-aware p=4 patchify
  -> stable coarse pair stream
  -> [pair-biased attention + outgoing/incoming triangle multiplication] x depth
  -> unpatchify + saved residue stream
  -> global residue attention
  -> atom decoder + saved atom stream + residue-local atom attention
  -> final Atom14 coordinates, 20-way amino-acid logits, compact distogram
```

The pair stream does **not** average four C-alpha coordinates into one point.
For every patch pair it retains all 16 ordered slot-to-slot C-alpha distances,
RBF-embeds them, and combines them with signed-log sequence separation,
same-chain, and validity features. Partial tail patches are masked exactly, so
chain lengths do not need to be divisible by four and patches never cross a
chain break.

There is no triangle attention, node-to-pair refresh, geometry refresh, pair
transition, irregular spatial-neighbor graph, or intermediate prediction head.
See [the architecture note](docs/architecture.md).

## Install

Python 3.10+ and PyTorch 2.5+ are required. Koochak is deliberately retained as
a pinned submodule for DDP, prefetch, BF16, EMA, checkpoints, and logging.

```bash
git clone --recurse-submodules git@github.com:jamaliki/hierarchical-kaveh.git
cd hierarchical-kaveh
python -m pip install -e external/koochak -e ".[dev,wandb,cuda]"
```

The ordinary correctness path works without custom FlashAttention. On H100,
install the exact patched FA3 build shipped as a patch in this repository:

```bash
./scripts/install_flash_attention.sh
```

The installer pins both upstream FlashAttention and CUTLASS, applies
`third_party/flash-attention/pair-bias.patch`, installs the Hopper package, and
verifies these symbols:

- `flash_attn_varlen_func`
- `flash_attn_pair_bias_func`
- `flash_attn_compact_pair_bias_func`

This is a custom extension; a normal `flash-attn` wheel is not sufficient for
the pair-bias FA3 path. Kernels select FA3 when its required symbol is present,
then Triton where supported, and otherwise use the eager/PyTorch reference path.
The fallback is meant for correctness and development, not H100 throughput.

## Train

Set `data.metadata_path` in `configs/train.yaml`, then run one process per GPU:

```bash
torchrun --standalone --nproc-per-node=8 \
  -m hierarchical_kaveh.train --config configs/train.yaml
```

Resume explicitly or from `train.out_dir/latest.pt`:

```bash
torchrun --standalone --nproc-per-node=8 \
  -m hierarchical_kaveh.train --config configs/train.yaml --resume latest
```

Restartable tasks should use the same immutable command on every attempt:

```bash
torchrun --standalone --nproc-per-node=8 \
  -m hierarchical_kaveh.train --config configs/train.yaml --resume auto
```

`auto` starts at step zero when no valid published numbered checkpoint exists;
otherwise Koochak selects the highest checkpoint with a valid ready manifest.
It never treats `latest.pt` or checkpoint scaffolding as resume evidence. Keep
`train.out_dir` and `wandb.name` (or `wandb.id`) stable across attempts and set
`wandb.resume: allow` for W&B-backed restartable tasks.

Training follows Pallatom's standard EDM contract: scaled log-normal noise,
rigid augmentation, stopped-gradient Kabsch-aligned coordinate MSE,
`1/c_out^2` weighting, 100% coordinate self-conditioning, Adam at `1e-3`
without a scheduler, crop size 128, batch size 32, and 300k steps. Supported
terminal objectives are coordinate denoising, 20-class amino-acid CE, smooth
all-atom lDDT, and the compact residue distogram. Koochak writes strict raw/EMA
checkpoints and optional stdout, CSV, JSONL, and W&B logs.

The source-by-source training audit, exact matches, and deliberate architectural
deviations are recorded in [the Pallatom conformance matrix](docs/pallatom_conformance.md).

The production YAML enables whole-model `torch.compile` and requires both the
compile and fused CUDA paths to succeed rather than silently falling back. Set
`train.compile.enabled`, `train.require_compile`, and `train.require_fused` to
`false` together for CPU development or correctness-only runs.

The reader consumes the existing Kaveh ragged Atom14 shards directly; see
[the data format](docs/data.md).

## Sample

Use the same config as training so the checkpoint's model configuration can be
verified exactly. One integer generates a monomer; comma-separated integers
generate multiple chains in one sample.

```bash
python -m hierarchical_kaveh.sample \
  --config configs/train.yaml \
  --checkpoint runs/hierarchical-kaveh/latest.pt \
  --lengths 256 \
  --num-samples 8 --batch-size 8 \
  --output-dir samples/l256

python -m hierarchical_kaveh.sample \
  --config configs/train.yaml \
  --checkpoint runs/hierarchical-kaveh/latest.pt \
  --lengths 96,96 \
  --output-dir samples/two_chain
```

Sampling implements Pallatom's perturbed-time stochastic Euler procedure with
the published `gamma=0.2`, noise scale `1.003`, step scale `2.25`, and two-pass
coordinate self-conditioning. Final sequence sampling uses temperature `0.1`.
Each sample produces a paired `.pdb` and `.fasta`.
EMA weights are required by default; `--raw` explicitly selects training
weights.

## Repository surface

```text
hierarchical_kaveh/
  config.py, types.py       strict shared contracts
  data/                     existing-shard reader and standard EDM batches
  diffusion/                corruption, schedules, and terminal losses
  model/                    the one hierarchical denoiser and kernels
  training.py, train.py     narrow Koochak training adapter and CLI
  sampling.py, sample.py    compact sampler and CLI
  io.py                     strict checkpoint and PDB/FASTA I/O
configs/train.yaml          train and sampling defaults
external/koochak/           pinned runtime submodule
third_party/flash-attention reproducible pair-bias FA3 patch
tests/                      CPU contracts plus optional CUDA parity tests
```

## Deliberate exclusions

This repository has no legacy Kaveh architecture, checkpoint migration,
Dunbrack dependency or rotamer augmentation, PPI/fixed-target path, motif
scaffolding, CFG, branching, secondary-structure/3Di/fold-class heads, dataset
builder, evaluation suite, sampling sweeps, or historical probes. Those features
are not hidden behind compatibility flags; they are outside this repository's
contract.

Pallatom's intermediate decoder supervision is intentionally absent because this
model emits only final predictions. Its local atomic-distogram auxiliary head is
also absent because the promoted atom stream has no persistent atom-pair tensor.
Neither omission is disguised as an equivalent coordinate loss.
