# JSC Baselines — протокол этапа 3

Этот контур сравнивает три нейросети Jarvis, обучаемые с нуля на одном корпусе:

- `char_cnn` — локальные символьные признаки через residual CNN;
- `bigru` — последовательный двунаправленный GRU encoder;
- `tiny_transformer` — self-attention encoder для дальних зависимостей.

У моделей одинаковые собственный символьный токенизатор, Transformer decoder,
JAL-цель, optimizer, sampling, лимиты и метрики. Это позволяет сравнивать именно
encoder, а не три несопоставимых pipeline. Hugging Face, готовые embeddings,
чужие weights и внешние AI API не используются.

## Что считается честным экспериментом

- vocabulary строится только по `train`;
- лучшая эпоха выбирается только по validation token NLL;
- test измеряется после выбора эпохи и не участвует в early stopping;
- `evaluation_holdout.jsonl` этот pipeline не открывает;
- запускаются три seed: 17, 29 и 41;
- итог сравнивается по mean/std exact JAL, schema validity и false execution;
- невалидный JSON получает ноль, даже если отдельные символы похожи на ответ;
- smoke-checkpoint проверяет только исправность кода и не является результатом.

## Команды на текущем компьютере

Проверить данные, CUDA и размеры моделей без создания файлов:

```powershell
.\training_workspace\START_JSC_BASELINES.ps1 -Device cuda -CheckOnly
```

Коротко проверить forward/backward/checkpoint каждой архитектуры:

```powershell
.\training_workspace\START_JSC_BASELINES.ps1 -Device cuda -Smoke
```

## Полный запуск на RTX 3090 Ti

После получения актуального проекта:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv-training\Scripts\Activate.ps1
.\training_workspace\START_JSC_BASELINES.ps1 -Device cuda -CheckOnly
.\training_workspace\START_JSC_BASELINES.ps1 -Device cuda
```

Будут последовательно выполнены девять runs: три архитектуры × три seed.
Checkpoint и отчёты сохраняются в `training_workspace/jsc_runs/`, которая
исключена из Git. После прерывания электричества или процесса:

```powershell
.\training_workspace\START_JSC_BASELINES.ps1 -Device cuda -ResumeExisting
```

`latest.pt` содержит model/optimizer/scheduler/AMP/RNG, поэтому продолжение не
начинает эксперимент заново. Итоговый `leaderboard.json` ранжирует модели только
по validation-метрикам. До завершения этого этапа ни один baseline не заменяет
текущую runtime NLU Jarvis.
