# Structured JSC без JSON — benchmark

## Итог

**Migration gates: FAIL.** Production routing не переключался. Выбран seed `17` только по validation; его migration Exact JAL — 38.25%.

Модель не содержит autoregressive decoder/token head и не генерирует JSON: она напрямую предсказывает act, число и порядок шагов, tools, аргументы, missing slots, reason и независимый execution verifier. JAL собирается типизированным schema validator в fail-closed режиме.

## Протокол

- Seeds: `[17, 29, 41]`; validation: 290; migration development: 400.
- Train: 2796 примеров; добавленные structured families создавались без чтения migration/test/holdout.
- Topology: d_model=64, encoder_layers=2, FFN=128; batch=64; lr=0.0008.
- Epoch checkpoint и confidence thresholds выбирались только на validation.
- Migration suite оценивался после выбора; приложения/tools не запускались.
- Locked `test` и `evaluation_holdout` не открывались.

## Результаты по seed

| Seed | Params | Epoch | Validation Exact | Migration Exact | Tool seq | Arguments | False execute | Opposite |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 245,435 | 2 | 28.62% | 38.25% | 51.00% | 47.50% | 0.00% | 0.00% |
| 29 | 245,435 | 4 | 27.59% | 39.75% | 53.50% | 50.00% | 0.00% | 0.00% |
| 41 | 245,435 | 1 | 27.93% | 37.50% | 50.25% | 46.75% | 0.00% | 0.00% |

Среднее Migration Exact JAL: **38.50% ± 1.15%**.

## Главные ошибки выбранного seed

- Wrong dialogue act: 218.
- Wrong arguments: 28.
- Wrong tool sequence: 1.
- Validation-selected thresholds: `{'execution_threshold': 0.65, 'verifier_threshold': 0.7, 'parameter_threshold': 0.35, 'span_threshold': 0.2, 'missing_threshold': 0.35}`.

## Обязательные gates выбранного кандидата

| Gate | Actual | Target | Status |
|---|---:|---:|:---:|
| overall_exact_jal_accuracy | 38.25% | 84.00% | FAIL |
| single_exact_jal_accuracy | 20.00% | 87.00% | FAIL |
| steps_2_3_exact_jal_accuracy | 31.40% | 85.00% | FAIL |
| steps_4_5_exact_jal_accuracy | 32.50% | 82.00% | FAIL |
| multi_turn_exact_jal_accuracy | 100.00% | 87.00% | PASS |
| schema_valid_rate | 100.00% | 100.00% | PASS |
| maximum_false_execution_rate | 0.00% | 0.20% | PASS |
| maximum_opposite_action_rate | 0.00% | 0.00% | PASS |

## Сравнение с предыдущим этапом

- Production NLU reference: 18.50% Exact JAL.
- JSC v8 structured-head reference: 16.75%.
- Новый Structured JSC, выбранный seed: 38.25%.
- Предыдущий direct Structured JSC (`9753746`): 12.00%; рост выбранного кандидата: +26.25%.

## Решение

`keep_current_production_routing`. До прохождения всех gates новый checkpoint остаётся экспериментальным; production wiring не изменён.
