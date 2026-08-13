# JSC v8 Data-first Scaling Curve

## Протокол

- Полная топология JSC v8 обучается end-to-end с нуля; staged v8 checkpoint не используется.
- Поднаборы вложены, выбираются целыми structural families и балансируют category/act/tool/step-count.
- Для приблизительно сопоставимого optimisation budget число эпох масштабируется обратно запрошенной доле. Из-за неделимых families фактический budget выше полного на 12.9% / 5.4% / 2.7% для первых трёх точек; это даёт малым наборам небольшое преимущество и не может искусственно создать наблюдаемое plateau.
- Seeds: `[17, 29, 41]`; locked test/holdout не открывались.
- Главная метрика — Structured JSC без autoregressive JSON.
- `selected` — checkpoint по composite validation NLL; `latest` — состояние в момент early stop, показанное как post-hoc diagnostic, а не новый выбранный кандидат.

## Кривая

| Доля | Примеры | Families | Validation Exact | Migration selected | Migration latest | Single | 2–3 шага | 4–5 шагов | Multi-turn | False execute |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25% | 395 | 84 | 7.47% ± 1.05% | 10.25% ± 1.15% | 10.83% ± 2.36% | 0.42% | 6.98% | 2.08% | 0.00% | 0.00% |
| 50% | 738 | 174 | 7.01% ± 0.20% | 10.17% ± 0.80% | 10.83% ± 2.55% | 10.42% | 3.88% | 0.00% | 0.00% | 0.00% |
| 75% | 1078 | 307 | 9.66% ± 0.60% | 11.83% ± 1.53% | 13.25% ± 1.64% | 11.25% | 0.78% | 0.00% | 0.00% | 0.00% |
| 100% | 1400 | 467 | 11.38% ± 0.69% | 12.67% ± 0.52% | 14.75% ± 1.09% | 12.92% | 0.39% | 0.42% | 0.00% | 0.00% |

## Где находится потолок v8

| Метрика на 100% | Среднее | Std |
|---|---:|---:|
| Migration dialogue act | 22.33% | 1.53% |
| Migration tool sequence | 26.00% | 1.32% |
| Migration arguments | 23.50% | 0.66% |
| Tool head, train at selected epoch | 34.12% | 5.62% |
| Tool head, validation at selected epoch | 24.60% | 6.38% |
| Span head, train at selected epoch | 54.75% | 3.19% |
| Span head, validation at selected epoch | 3.03% | 0.81% |

## Диагноз

- Verdict: `plateau_architecture_or_objective_limited`.
- Рост 25% → 100%: +2.42%.
- Рост 75% → 100%: +0.83%.
- Максимальное standard deviation между seed: 1.53%.
- Монотонный рост: `False`.
- Диагностический latest растёт монотонно (`True`), но всего на +3.92% от 25% до 100% и на +1.50% от 75% до 100%; это не меняет verdict.
- Multi-turn остался на 0%; планы на 2–3 шага ухудшились до 0.39%, на 4–5 шагов — 0.42%.
- Нулевые false-execute/opposite-action достигаются в основном fail-closed отказами; это безопасно, но не означает готовность выполнять команды.
- Data-first улучшает отдельные классификационные головы, но не преодолевает рассогласование JSON reconstruction, tool sequence и span extraction.

## Решение

1. Не расходовать следующий цикл на простое добавление похожих synthetic данных в v8.
2. Перейти к Structured JSC без JSON: прямой act/count/tool/argument/span decoder, отдельные loss schedules и checkpoint selection по program-level метрикам.
3. Использовать текущие данные как baseline, затем расширять только доказанно слабые families: multi-turn, 2–5 действий, ASR aliases и свободные аргументы.
4. Не открывать locked test/holdout, пока новый structured-кандидат не пройдёт development migration gates.
