# Inference validation

The adapter follows the release-pinned deployment and training revisions in
`native-sources.json`. The author actor and backward encoder are used unchanged.

```bash
uv run --with torch validate_inference.py --help
```

Run the validator with `uv run --with torch validate_inference.py`, passing
`--source`, `--weights`, `--encoder` and an `--output` outside Git. The source
folder contains `ufo-release-source` and `ufo-training-source` checked out at their
pinned commits, plus `ufo-artifacts/exported/FBcprAuxModel.meta.json` from the pinned
Hugging Face release. Every source file used is hash-checked before execution.

The check calls the author's reference-observation functions, observation-history
classes and complete policy step on identical input sequences. It covers 160
states, four resets, joint reordering, near-vertical rotations, action clipping,
target slew limits and the final reference windows.

The policy consumes three future latents at 20/40/60 ms. The 80 ms query only
detects the end timestamp and does not enter the encoder. Native exported windows
exclude the exact last timestamp and shrink near the end. After the last exported
context, the adapter holds it through the benchmark's exact final timestamp;
it never loops the motion or shortens the rollout. Simulation, initialization and
reference FK are the shared benchmark's, not the author's training environment.
