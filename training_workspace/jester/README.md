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
4. Run an equal-budget, class-balanced five-epoch benchmark of
   `tiny_3d_cnn`, `cnn_bigru`, and `mobilenet_tsm_attention`.
5. Select by validation macro-F1. Negative-class recall is reported separately.
6. Train the selected random-initialized model on full train only; validation
   selects the checkpoint and early stopping. Test is never loaded here.
7. Open official test exactly once. A preserved final report prevents an
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
.\.venv-jester\Scripts\python.exe -m src.jester.training benchmark
.\.venv-jester\Scripts\python.exe -m src.jester.training train
$winner = (Get-Content reports/jester/benchmark.json | ConvertFrom-Json).recommended_winner
.\.venv-jester\Scripts\python.exe -m src.jester.evaluate --checkpoint "training_workspace/jester/runs/full/$winner/best.pt"
.\.venv-jester\Scripts\python.exe -m src.jester.export --checkpoint "training_workspace/jester/runs/full/$winner/best.pt"
```

For a second Windows PC, install the latest NVIDIA driver and 64-bit Python
3.10, then clone the prepared branch:

```powershell
git clone --branch codex/checkpoint-2026-08-09 --single-branch https://github.com/Pac1FF1zM/Jarvis_AI.git
cd Jarvis_AI
```

After making your licensed dataset available locally while keeping it under
your control, the intended order is:

```powershell
PREPARE_JESTER_TRAINING.cmd
START_JESTER_BENCHMARK.cmd
START_JESTER_TRAINING.cmd
```

Training writes `latest.pt` atomically after every epoch and resumes it by
default after interruption. Pass `--fresh` only when intentionally starting a
new run with the same output directory.

On a new high-end PC, generate and use a local untracked hardware profile:

```powershell
.\.venv-jester\Scripts\python.exe -m src.jester.preflight --workers 0,4,8,12,16 --write-profile configs/jester_hardware.yaml
.\.venv-jester\Scripts\python.exe -m src.jester.rehearsal --config configs/jester_hardware.yaml
.\.venv-jester\Scripts\python.exe -m src.jester.quality_gate --config configs/jester_hardware.yaml
.\.venv-jester\Scripts\python.exe -m src.jester.doctor --config configs/jester_hardware.yaml
.\.venv-jester\Scripts\python.exe -m src.jester.training benchmark --config configs/jester_hardware.yaml
```

## Verified laptop profile

The final preparation audit on the RTX 3050 6 GB selected batch size 32 and
eight Windows DataLoader workers. Real-JPEG throughput was about 78.3
clips/second with non-persistent worker pools.
Peak reserved VRAM at batch 32 was 1.26 GB (`tiny_3d_cnn`), 1.31 GB
(`cnn_bigru`), and 1.45 GB (`mobilenet_tsm_attention`). The configured winner
is intentionally empty: full training reads the winner from the completed
candidate benchmark instead of assuming one in advance.

Windows workers are intentionally non-persistent. This prevents train and
validation worker pools from coexisting and exhausting the Windows commit/page
file when every spawned process loads the CUDA-enabled PyTorch runtime.

Do not run the final evaluation until architecture and hyperparameters are
frozen. Do not mix, distribute, or ship derived weights in Jarvis without a
separate license review.
