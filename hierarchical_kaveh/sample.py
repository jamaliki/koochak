"""Command-line unconditional sampling to paired PDB and FASTA files."""

from __future__ import annotations

import argparse

import torch

from .config import load_config
from .io import load_checkpoint, write_sample_batch
from .model import HierarchicalKaveh
from .sampling import parse_chain_lengths, sample


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="The strict run YAML used for training.")
    parser.add_argument("--checkpoint", required=True, help="A Koochak checkpoint.")
    parser.add_argument(
        "--lengths",
        required=True,
        help="One length for a monomer, or comma-separated chain lengths.",
    )
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", default="./samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--raw", action="store_true", help="Use raw rather than EMA weights.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("num-samples and batch-size must be positive")
    device = torch.device(args.device)
    config = load_config(args.config)
    chain_lengths = parse_chain_lengths(args.lengths)
    model = HierarchicalKaveh(config.model).to(device)
    load_checkpoint(model, args.checkpoint, config=config, use_ema=not args.raw)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    completed = 0
    while completed < args.num_samples:
        current_batch = min(args.batch_size, args.num_samples - completed)
        result = sample(
            model,
            chain_lengths,
            batch_size=current_batch,
            config=config.sampling,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        write_sample_batch(
            args.output_dir,
            result.coordinates,
            result.aatype,
            chain_lengths,
            start_index=completed,
        )
        completed += current_batch


if __name__ == "__main__":
    main()
