# Structured JSC без JSON — benchmark

## Итог

**Migration gates: PASS.** Production routing не переключался. Выбран seed `29` только по validation; его migration Exact JAL — 84.75%.

Модель не содержит autoregressive decoder/token head и не генерирует JSON: она напрямую предсказывает act, число и порядок шагов, tools, аргументы, missing slots, reason и независимый execution verifier. JAL собирается типизированным schema validator в fail-closed режиме.

## Протокол

- Seeds: `[29]`; validation: 290; migration development: 400.
- Train: 4355 примеров; добавленные structured families создавались без чтения migration/test/holdout.
- Topology: d_model=96, encoder_layers=2, FFN=192; batch=32; lr=0.00035.
- Epoch checkpoint и confidence thresholds выбирались только на validation.
- Migration suite оценивался после выбора; приложения/tools не запускались.
- Locked `test` и `evaluation_holdout` не открывались.

## Результаты по seed

| Seed | Params | Epoch | Validation Exact | Migration Exact | Tool seq | Arguments | False execute | Opposite |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 29 | 534,942 | 5 | 35.52% | 84.75% | 94.25% | 90.75% | 0.00% | 0.00% |

Среднее Migration Exact JAL: **84.75% ± 0.00%**.

## Главные ошибки выбранного seed

- Wrong dialogue act: 30.
- Wrong arguments: 31.
- Wrong tool sequence: 0.
- Validation-selected thresholds: `{'execution_threshold': 0.65, 'verifier_threshold': 0.9, 'parameter_threshold': 0.35, 'span_threshold': 0.25, 'missing_threshold': 0.35}`.

## Обязательные gates выбранного кандидата

| Gate | Actual | Target | Status |
|---|---:|---:|:---:|
| overall_exact_jal_accuracy | 84.75% | 84.00% | PASS |
| single_exact_jal_accuracy | 90.00% | 87.00% | PASS |
| steps_2_3_exact_jal_accuracy | 93.02% | 85.00% | PASS |
| steps_4_5_exact_jal_accuracy | 100.00% | 82.00% | PASS |
| multi_turn_exact_jal_accuracy | 100.00% | 87.00% | PASS |
| schema_valid_rate | 100.00% | 100.00% | PASS |
| maximum_false_execution_rate | 0.00% | 0.20% | PASS |
| maximum_opposite_action_rate | 0.00% | 0.00% | PASS |

## Сравнение с предыдущим этапом

- Production NLU reference: 18.50% Exact JAL.
- JSC v8 structured-head reference: 16.75%.
- Новый Structured JSC, выбранный seed: 84.75%.
- Предыдущий direct Structured JSC (`9753746`): 12.00%; рост выбранного кандидата: +72.75%.

## Решение

`eligible_for_shadow_wiring_review`. До прохождения всех gates новый checkpoint остаётся экспериментальным; production wiring не изменён.
