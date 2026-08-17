# Reproduce the README character matrix

This directory contains the complete input and prompt package used for the
README matrix. It deliberately contains exactly three distinct OpenPose maps
(standing, dance and jump) and one Depth map (two thumbs up).

## Included

- `demo.json`: exact prompts, negative prompts, seeds, batch sizes, selected
  indices, strengths, model filenames, byte sizes and SHA-256 hashes.
- `sources/`: the four control inputs used by the API.
- `scripts/generate.py`: standard-library Forge API client.
- `scripts/validate.py`: strict source, metadata, anti-copy and background
  consistency gates; writes its evidence to `outputs/quality-report.json`.
- `scripts/compose.py`: deterministic 1740×1510 README montage.

The generated PNG files and reports under `outputs/` are intentionally ignored
by Git. The final curated montage is tracked at
`../../assets/forge-character-control-matrix.webp`.

The API PNGs remain untouched under `outputs/raw`. For presentation only, the
validator and composer isolate each Depth subject and place it on the exact same
RGB 163/160/146 background. The operation is implemented in
`scripts/image_ops.py`, and normalized panels are written to `outputs/final`.

## Requirements

Use sd-webui-forge-classic 2.28.1 (Neo), this extension, Python with Pillow and
NumPy, and the five model files listed in `demo.json`. Their full hashes are
recorded so a different checkpoint or module cannot silently pass as the demo
environment.

Start Forge with `--api`, select the checkpoint, VAE and text encoder recorded
in the manifest, then run from this directory:

```powershell
python scripts/generate.py --dry-run
python scripts/generate.py --verify-models --forge-root G:\sd-webui-forge-classic-2.28.1 --dry-run
python scripts/generate.py --api-url http://127.0.0.1:7860
python scripts/validate.py --output-dir outputs/raw --presentation-dir outputs/final
python scripts/compose.py --input-dir outputs/raw
```

Use `--run RUN_ID` repeatedly to regenerate only selected runs. Keep each
recorded batch size and selected index: several dance/jump panels came from one
alternating two-reference batch, so replacing that grouped run with independent
single-image requests is not the same experiment.

## Acceptance criteria

The automated validator fails unless:

- the OpenPose corpus contains exactly three unique hashes;
- all 16 selected outputs exist, are distinct and are 768×768;
- API metadata proves the expected control mode;
- OpenPose source-colour and black-background copying stay below fixed limits;
- Depth outputs are not literal copies of the input;
- all four Depth corner-background colours stay within the fixed pairwise
  distance limit.

The final visual review also checks two arms on Bart standing, canonical Bart
proportions and four-finger hands, two readable thumbs on every Depth result,
complete gloves for Deadpool, Black Widow and Darth Vader, Black Widow's black
tactical suit, complete limbs, and absence of coloured OpenPose sticks.

Character names and designs belong to their respective rights holders. The
control maps and prompts are provided only as a technical extension demo; no
external character reference image is required or included.
