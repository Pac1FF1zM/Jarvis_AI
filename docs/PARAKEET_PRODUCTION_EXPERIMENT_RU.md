# Experimental production Parakeet — отчёт 2026-08-13

## Что включено

`nvidia/parakeet-tdt-0.6b-v3` revision
`541d1f99c6b0c3cd0b11a95167540bb8edefd82b` подключён к основному STT-контуру
Jarvis. Модель остаётся в изолированном persistent worker и не импортируется в
процесс EventBus. Production-включение требует двух явных параметров:

```yaml
modules:
  stt:
    params:
      engine: parakeet
      experimental_production: true
```

Откат выполняется заменой `engine: parakeet` на `engine: whisper`. Скрытого
fallback нет: это сохраняет чистоту A/B-наблюдений и не маскирует сбой одного
движка результатом другого.

## Результат парного smoke benchmark

Обе модели получили одни и те же 20 записей из публичного validation split
FLEURS `ru_ru/dev`. Все записи содержат человеческую речь, имеют human-authored
reference и не превышают production-лимит Jarvis в 12 секунд. Warm-up и загрузка
моделей исключены из decode latency. Модели запускались последовательно на
NVIDIA GeForce RTX 3050 Laptop GPU 6 ГБ.

| Метрика | Parakeet TDT 0.6B v3 | Whisper small |
|---|---:|---:|
| Corpus WER | **4,40%** | 5,35% |
| Corpus CER | 2,16% | **1,83%** |
| Exact match | **70%** | 50% |
| Mean decode | **1210 мс** | 2149 мс |
| Median decode | **1219 мс** | 2094 мс |
| p95 decode | **1802 мс** | 2970 мс |
| Real-time factor | **0,133** | 0,237 |
| Model load | 6847 мс | **4718 мс** |

На этой малой выборке Parakeet лучше по WER и примерно в 1,78 раза быстрее по
средней decode latency. Whisper немного лучше по CER, а его модель загружается
быстрее. Результат является основанием для experimental production A/B, но не
для окончательного удаления Whisper: FLEURS содержит общий русский текст, а не
короткие команды Jarvis, и 20 записей недостаточно для финального решения.

## Обнаруженная и исправленная проблема

Первый прогон выявил обрезание русских гипотез старым лимитом 40 generated
tokens: WER Parakeet был 40,76%. После увеличения bound до 96 при неизменном
12-секундном audio limit обрезание исчезло, а на исходной выборке WER снизился до
6,52%. Это изменение покрыто regression-тестом.

## Проверки

- real CUDA model smoke: `1 passed`;
- STT/Parakeet/runtime diagnostics: `56 passed, 1 skipped`;
- `main.py --demo`: worker загрузился одновременно с Gesture Core, основной
  EventBus получил `transcription_ready`, shutdown завершил worker чисто;
- Runtime Doctor: Parakeet interpreter и закреплённый snapshot найдены.

Полный per-item JSON находится локально в
`reports/parakeet_vs_whisper_fleurs_ru_production_20.json`. Публичный smoke не
использовал и не создавал приватные записи владельца.
