# Jester from-scratch pipeline

This workspace is isolated from the runtime and from the existing IPN
checkpoint. Raw data, runs and exported weights are ignored by Git. The
official Qualcomm metadata is used without changing class names or split
membership.

## Fixed protocol

1. Download all three official `20bn-jester-v1-00..02` parts and labels.
2. Hash every archive part and stream-extract the multipart TGZ without making
   a second 22.8 GB combined archive.
3. Build one manifest: official train -> `train`, validation -> `val`, and
   test answers -> sealed `test`.
4. The equal-budget benchmark completed on 2026-08-09. `tiny_3d_cnn` won by
   validation macro-F1 (`0.1992`, versus `0.0368` and `0.0254`) and is now the
   only configured training model.
5. Train `tiny_3d_cnn` from random initialization on full train only; validation
   selects the checkpoint and early stopping. Test is never loaded here.
6. Open official test exactly once. A preserved final report prevents an
   accidental second evaluation. Any downstream use or export remains
   research-only and requires a separate license review.

## License gate

Downloading Jester constitutes acceptance of Qualcomm's Research Use License.
It permits internal non-profit research use, prohibits commercial use, and
places restrictions on distribution and combining the dataset or derived
results with third-party data. Review the
[official license](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/jester_something_something_exercise_research_license_final_qti_28jul2022.pdf)
before continuing. The downloader will refuse to start without an explicit
acceptance flag and records that acceptance locally in ignored raw-data space.

## Commands

```powershell
SETUP_JESTER_TRAINING.cmd

.\.venv-jester\Scripts\python.exe -m src.jester.download --accept-research-license
.\.venv-jester\Scripts\python.exe -m src.jester.acquire
.\.venv-jester\Scripts\python.exe -m src.jester.prepare
.\.venv-jester\Scripts\python.exe -m src.jester.doctor
.\.venv-jester\Scripts\python.exe -m src.jester.smoke
.\.venv-jester\Scripts\python.exe -m src.jester.preflight
.\.venv-jester\Scripts\python.exe -m src.jester.rehearsal
.\.venv-jester\Scripts\python.exe -m src.jester.quality_gate
.\.venv-jester\Scripts\python.exe -m src.jester.training train --model tiny_3d_cnn
.\.venv-jester\Scripts\python.exe -m src.jester.evaluate --checkpoint "training_workspace/jester/runs/full/tiny_3d_cnn/best.pt"
.\.venv-jester\Scripts\python.exe -m src.jester.export --checkpoint "training_workspace/jester/runs/full/tiny_3d_cnn/best.pt"
```

For a second Windows PC, install the latest NVIDIA driver and 64-bit Python
3.10, then clone the prepared branch:

```powershell
git clone --branch codex/checkpoint-2026-08-09 --single-branch https://github.com/Pac1FF1zM/Jarvis_AI.git
cd Jarvis_AI
```

The dataset and generated checkpoints are intentionally absent from GitHub.
Keep them under the license holder's control. Either run the licensed
downloader on the second PC or copy these prepared local paths without adding
them to Git:

```text
data/raw/jester/downloads/
data/raw/jester/metadata/jester_labels/
data/raw/jester/frames/20bn-jester-v1/
data/raw/jester/RESEARCH_LICENSE_ACCEPTED.json
data/splits/jester/manifest.jsonl
```

The intended order on the second PC is now only:

```powershell
PREPARE_JESTER_TRAINING.cmd
START_JESTER_TRAINING.cmd
```

Training writes `latest.pt` atomically after every epoch and resumes it by
default after interruption. The resumable checkpoint is:

```text
training_workspace/jester/runs/full/tiny_3d_cnn/latest.pt
```

To continue on another PC, copy the complete
`training_workspace/jester/runs/full/tiny_3d_cnn/` directory and the same
licensed dataset. Machine-local paths, DataLoader workers, CUDA micro-batch and
VRAM safety settings may differ; seed, effective batch, optimizer settings,
frame shape and exact train/validation membership must remain unchanged. Pass
`--fresh` only when intentionally starting a new run with the same output
directory.

On a new high-end PC, generate and use a local untracked hardware profile:

```powershell
.\.venv-jester\Scripts\python.exe -m src.jester.preflight --workers 0,4,8,12,16 --write-profile configs/jester_hardware.yaml
.\.venv-jester\Scripts\python.exe -m src.jester.rehearsal --config configs/jester_hardware.yaml
.\.venv-jester\Scripts\python.exe -m src.jester.quality_gate --config configs/jester_hardware.yaml
.\.venv-jester\Scripts\python.exe -m src.jester.doctor --config configs/jester_hardware.yaml
.\.venv-jester\Scripts\python.exe -m src.jester.training train --config configs/jester_hardware.yaml --model tiny_3d_cnn
```

## Verified laptop profile

The completed laptop benchmark used batch size 32. The current preparation
step chooses a RAM-safe worker count for the active PC and probes only
`tiny_3d_cnn`. Peak reserved VRAM for Tiny3D was about 1.26 GB at batch 32 on
the RTX 3050 6 GB. The winner is fixed in the tracked config, so a cloned PC
does not need the ignored local benchmark report.

Windows workers are intentionally non-persistent. This prevents train and
validation worker pools from coexisting and exhausting the Windows commit/page
file when every spawned process loads the CUDA-enabled PyTorch runtime.

Do not run the final evaluation until architecture and hyperparameters are
frozen. Do not mix, distribute, or ship derived weights in Jarvis without a
separate license review.
