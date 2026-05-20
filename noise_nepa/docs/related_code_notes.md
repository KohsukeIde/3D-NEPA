# Related code/repo notes

Before creating this track, we checked the public code layout and stated training interfaces of the closest codebases.

## PointDif

PointDif uses a conventional 3D pretraining repository layout: `cfgs/`, `data/`, `datasets/`, `models/`, `tools/`, `utils/`, `pointdif_main.py`, and scripts through config files. It trains PointDif with `pointdif_main.py --config cfgs/pretrain.yaml` and fine-tunes with `--finetune_model`.

Implementation implication for Noise-NEPA:
- keep a separate `configs/`, `data/`, `models/`, `train/`, `eval/`, `scripts/` layout;
- expose all smoke experiments through shell wrappers and Python entrypoints;
- avoid mutating the existing PointGPT code in the first implementation.

## Point-MaDi

Point-MaDi also uses `cfgs/`, `datasets/`, `models/`, `scripts/`, `tools/`, plus downstream folders for part and semantic segmentation. Its README exposes pretrained/fine-tune configs and downstream protocols.

Implementation implication:
- keep smoke pretraining and evaluation scripts separated;
- plan downstream transfer only after pre-mortem and internal dynamics pass;
- do not present internal denoising retrieval as a downstream benchmark.

## PointGPT / 3D-NEPA PointGPT

PointGPT remains the main backbone family for this repo. Noise-NEPA starts with a simple encoder for robust smoke tests and includes a best-effort `PointGPTFeatureAdapter` for later integration.

Implementation implication:
- first test if diffusion-time latent dynamics exists at all;
- only then integrate PointGPT features / ScanObjectNN transfer.
