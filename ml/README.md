# Jarvis ML baseline

Следующее поколение собственной NLU развивается как Jarvis Semantic Core.
Рабочий порядок и гейты: [`JSC_ROADMAP_RU.md`](JSC_ROADMAP_RU.md).

This directory contains the project's own NLU pipeline. It uses no Hugging
Face components, pretrained weights or embeddings, downloaded tokenizers, or
external datasets. PyTorch is only the tensor/autograd framework.

The current task is multi-task Russian command understanding:

- intents: time, reminder, open application, list applications, cancel,
  general chat, and unknown;
- BIO slots: `duration`, `reminder_text`, and `application`.

The repository includes character and word tokenizers plus CharCNN and BiGRU
architectures, all initialized from random weights.

## Reproducible experiments

```powershell
python -m ml.nlu.train --architecture word_bigru --method standard --epochs 30 --batch-size 64 --hidden-dim 64 --max-length 32 --output models/nlu_word_bigru_standard.pt
python -m ml.nlu.train --architecture word_bigru --method curriculum --epochs 40 --batch-size 64 --hidden-dim 64 --max-length 32 --output models/nlu_word_bigru_curriculum.pt
python -m ml.nlu.train --architecture char_cnn --method augmented --epochs 25 --output models/nlu_cnn_augmented.pt
```

`standard`, `augmented`, and `curriculum` are independent training regimes.
Train, validation, and test use different exact phrase families. Every run
stores the seed, vocabulary, configuration, weights, and metrics.

## Selected model and honest boundary

`Word-BiGRU + curriculum` is selected by validation macro-F1: 0.908, versus
0.802 for standard and 0.662 for CharCNN + augmented. Scalar temperature
scaling improves validation ECE from 0.135 to 0.118.

- application launch: 21/24;
- application-list request: 4/4;
- reminder: 27/27;
- time requests: 4/4;

Development-test macro-F1 is 0.818. A separately frozen 49-example holdout,
protected by a SHA-256 sidecar and never read by training, scores 0.816 intent
accuracy, 0.818 macro-F1, and 0.816 exact frame accuracy. Its nine failures
were not fed back into training. Raw neural slot tagging remains incomplete
(development-test entity-token F1 0.670). Runtime therefore uses a constrained decoder for reminder/application
parameters. Application launching additionally requires an explicit imperative
phrase and a strict tool allow-list. Talking about a browser or calculator
cannot launch it accidentally.

Run `python -m ml.nlu.evaluate_holdout --checkpoint
models/nlu_word_bigru_curriculum.pt` to reproduce the frozen evaluation.
Improving OOD rejection and pure neural slot exact-match remains the next ML
milestone; future tuning requires a new holdout rather than training on this one.

## Manager training workspace

The original Word-BiGRU above remains a reproducible fallback. The current
GPU workspace trains a new CharCNN from scratch with an auxiliary hierarchical
route objective (`tool`, `control`, `dialogue`, `reject`), source-balanced
sampling, and three independent training regimes. Selection cannot export a
checkpoint unless it improves the new corpus, preserves the legacy regression
set, meets worst-class recall and CPU latency gates, and passes two holdouts.
See `training_workspace/GUIDE_RU.md` for the exact workflow.
