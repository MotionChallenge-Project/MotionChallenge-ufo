# UFO FB-CPR-Aux 192M

[Author project](https://roboparty.github.io/UFO/) · [Checkpoint](https://huggingface.co/xuewang/ufo-g1-policy/tree/da3727034a058eae588a16af142764ee34bdada4) · [Provenance](provenance.json)

## Install and run

```bash
git clone --recurse-submodules https://github.com/MotionChallenge-Project/MotionChallenge-ufo.git
cd MotionChallenge-ufo
uv sync
uv run evaluate.py
```

One complete rollout produces success, completion and six MPBPE values in a
verified `result.json`. MuJoCo 3.17 CPU batch; `--jobs 8` sets parallelism.
Missing data and both author weights are downloaded and SHA256-checked.
Results and traces stay outside Git.

Use cached artifacts with `--weights PATH --artifact encoder=PATH --data PATH --offline`.
The root `motion-challenge` submodule owns data, physics, scoring and verification.

[Inference validation](validation/README.md) · [Native parity report](validation/native-parity.json)
