# Parakeet TDT: experimental production STT and shadow diagnostic

The production STT module can now send each 16 kHz mono microphone capture to
the same isolated persistent worker. This path is enabled only when both
`engine: parakeet` and `experimental_production: true` are present in
`config.yaml`. It publishes the ordinary Jarvis `transcription_ready` event;
therefore `main.py` is a real action-capable runtime.

The separate shadow diagnostic records one capture in memory, sends it
to a persistent local Parakeet worker, and shows the production Jarvis NLU
interpretation as JSON. It has no EventBus, tool registry, publisher, or action
executor. Every action in its output is marked `execution: blocked`.

Pinned model:

- `nvidia/parakeet-tdt-0.6b-v3`
- revision `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`
- license `CC-BY-4.0`
- preferred provider `cuda`, FP16

The model card lists Linux as the preferred/supported OS. This Windows path
therefore uses the official Transformers implementation as an experimental
compatibility test and fails closed if CUDA or model loading is unavailable.

## Setup

From the repository root:

```bat
SETUP_PARAKEET.cmd --runtime
SETUP_PARAKEET.cmd --review-license
```

Read these local review files before accepting the model terms:

- `.local/parakeet/license-review/CC-BY-4.0.txt`
- `.local/parakeet/license-review/MODEL_CARD.md`

Acceptance must be an explicit owner action and is immutable:

```bat
SETUP_PARAKEET.cmd --accept-license CC-BY-4.0
SETUP_PARAKEET.cmd --download
SETUP_PARAKEET.cmd --status
```

Model-license evidence is stored separately from private fixture consent and
retention metadata.

## Live no-action test

```bat
TEST_PARAKEET_NLU.cmd --mic
```

Press Enter to start, speak normally, and press Enter again to finish. The
first startup loads and warms the model; later phrases reuse the same worker.

Other safe modes:

```bat
TEST_PARAKEET_NLU.cmd --list-devices
TEST_PARAKEET_NLU.cmd --mic --device 1
TEST_PARAKEET_NLU.cmd --wav "C:\absolute\sample.wav"
TEST_PARAKEET_NLU.cmd --text "открой калькулятор"
```

Use `TEST_PARAKEET_NLU.cmd` when actions must remain impossible. Use `main.py`
only for an intentional production A/B session, because commands that pass the
normal semantic/NLU/tool safety gates may execute. Roll back by setting
`modules.stt.params.engine: whisper`.

## Paired benchmark

`benchmarks/compare_stt.py` runs Parakeet and official Whisper sequentially on
the same JSONL manifest and reports corpus WER/CER, exact match, latency and
real-time factor. It contains no executor and never publishes to EventBus:

```bat
venv\Scripts\python.exe -m experiments.parakeet.benchmarks.compare_stt ^
  --manifest .local\parakeet\benchmarks\manifest.jsonl ^
  --provider cuda --whisper-model small ^
  --output reports\parakeet_vs_whisper.json
```

The initial public human-speech smoke used 20 FLEURS `ru_ru/dev` clips no longer
than Jarvis's 12-second capture limit. Parakeet achieved WER 4.40%, mean decode
1210 ms and RTF 0.133; Whisper small achieved WER 5.35%, mean decode 2149 ms and
RTF 0.237. This small general-Russian sample supports production testing but is
not a substitute for owner-approved command fixtures.

## Fixture status at the checkpoint

The repository contains the canonical `jarvis.semantic_fixture.v1` schema,
validator, plan generator and local recorder. The expected private fixture root
`.local/parakeet/fixtures/phase_1_5` did not exist at the 2026-08-13 checkpoint:
no plan was regenerated, no consent was requested, no audio was recorded and no
benchmark was started. Do not treat the presence of recorder code as
`READY_TO_RECORD` evidence.
