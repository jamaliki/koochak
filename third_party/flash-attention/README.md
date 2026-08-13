# FlashAttention pair-bias patch

Hierarchical Kaveh's fastest H100 coarse-attention path extends FlashAttention
with dense and compact pair-bias entry points. `pair-bias.patch` contains only
that extension, based on the upstream and CUTLASS commits recorded in
`VERSION`. The repository installer clones upstream, checks out those exact
commits, applies the patch, and verifies the three runtime symbols used here.

The patch is derived from FlashAttention and is distributed under its BSD
3-Clause license, reproduced in `LICENSE`.
