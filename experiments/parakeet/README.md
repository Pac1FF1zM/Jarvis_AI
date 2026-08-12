# Parakeet TDT + production NLU shadow diagnostic

This experiment records one 16 kHz mono microphone capture in memory, sends it
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

Do not run `main.py` to test this experiment. Parakeet is not wired into the
production STT module on this branch.

## Fixture status at the checkpoint

The repository contains the canonical `jarvis.semantic_fixture.v1` schema,
validator, plan generator and local recorder. The expected private fixture root
`.local/parakeet/fixtures/phase_1_5` did not exist at the 2026-08-13 checkpoint:
no plan was regenerated, no consent was requested, no audio was recorded and no
benchmark was started. Do not treat the presence of recorder code as
`READY_TO_RECORD` evidence.
