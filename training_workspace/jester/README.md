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
4. Run an equal-budget, class-balanced two-epoch benchmark of
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
.\.venv-jester\Scripts\python.exe -m src.jester.training benchmark
.\.venv-jester\Scripts\python.exe -m src.jester.training train --model mobilenet_tsm_attention
.\.venv-jester\Scripts\python.exe -m src.jester.evaluate --checkpoint training_workspace/jester/runs/full/mobilenet_tsm_attention/best.pt
.\.venv-jester\Scripts\python.exe -m src.jester.export --checkpoint training_workspace/jester/runs/full/mobilenet_tsm_attention/best.pt
```

Do not run the final evaluation until architecture and hyperparameters are
frozen. Do not mix, distribute, or ship derived weights in Jarvis without a
separate license review.
