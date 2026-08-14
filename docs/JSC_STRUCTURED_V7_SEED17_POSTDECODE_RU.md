# Structured JSC без JSON — benchmark

## Итог

**Migration gates: FAIL.** Production routing не переключался. Выбран seed `17` только по validation; его migration Exact JAL — 80.50%.

Модель не содержит autoregressive decoder/token head и не генерирует JSON: она напрямую предсказывает act, число и порядок шагов, tools, аргументы, missing slots, reason и независимый execution verifier. JAL собирается типизированным schema validator в fail-closed режиме.

## Протокол

- Seeds: `[17]`; validation: 290; migration development: 400.
- Train: 4355 примеров; добавленные structured families создавались без чтения migration/test/holdout.
- Topology: d_model=128, encoder_layers=3, FFN=256; batch=32; lr=0.0003.
- Epoch checkpoint и confidence thresholds выбирались только на validation.
- Migration suite оценивался после выбора; приложения/tools не запускались.
- Locked `test` и `evaluation_holdout` не открывались.

## Результаты по seed

| Seed | Params | Epoch | Validation Exact | Migration Exact | Tool seq | Arguments | False execute | Opposite |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 1,050,494 | 13 | 36.21% | 80.50% | 94.25% | 90.75% | 0.00% | 0.00% |

Среднее Migration Exact JAL: **80.50% ± 0.00%**.

## Главные ошибки выбранного seed

- Wrong dialogue act: 38.
- Wrong arguments: 40.
- Wrong tool sequence: 0.
- Validation-selected thresholds: `{'execution_threshold': 0.65, 'verifier_threshold': 0.8, 'parameter_threshold': 0.35, 'span_threshold': 0.35, 'missing_threshold': 0.35}`.

## Обязательные gates выбранного кандидата

| Gate | Actual | Target | Status |
|---|---:|---:|:---:|
| overall_exact_jal_accuracy | 80.50% | 84.00% | FAIL |
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
- Новый Structured JSC, выбранный seed: 80.50%.
- Предыдущий direct Structured JSC (`9753746`): 12.00%; рост выбранного кандидата: +68.50%.

## Решение

`keep_current_production_routing`. До прохождения всех gates новый checkpoint остаётся экспериментальным; production wiring не изменён.
