# Jarvis Semantic Core — актуальный roadmap

Актуально на 14 августа 2026. Полный runtime-контекст находится в
[`docs/PROJECT_STATUS_2026-08-14_RU.md`](../docs/PROJECT_STATUS_2026-08-14_RU.md).

## Цель

Построить собственный локальный semantic core, который из текста, истории и
типизированного dialogue state формирует проверяемый JAL-план и выбирает
`execute / ask / confirm / cancel / reject`. Все веса, tokenizer, data factory
и decoder принадлежат проекту; готовые language-model weights и внешние AI API
не используются.

JSC должен заменить production NLU только после независимых voice/safety gates.
До этого NLU остаётся execution routing, а JSC работает stateful shadow.

## Текущая архитектура

Structured JSC — компактный Transformer encoder с прямыми prediction heads:

- dialogue act;
- число и порядок шагов;
- segment boundaries и tool router;
- categorical arguments и character spans;
- missing slots и reason;
- независимый execution verifier.

Autoregressive decoder и генерация JSON отсутствуют. Heads собираются в JAL v1
через schema validator, semantic grounding и fail-closed structured decoder.
Runtime shadow передаёт history и pending JAL между ходами, но не публикует
исполняемые события.

## Выполненные этапы

1. **[готово] JAL v1 и tool contracts.** Канонический codec, typed schema,
   grammar validation, ordered plans и safety-oriented metrics.
2. **[готово] Reproducible data factory.** Independent family splits, single,
   compound, ASR, hard-negative, OOD, correction и multi-turn scenarios.
3. **[готово] Baselines.** CharCNN, BiGRU, tiny Transformer и direct structured
   candidates с одинаковым evaluation protocol.
4. **[готово] Structured heads без JSON.** Segmented router, parameters/spans,
   missing/reason/verifier heads, AMP-safe training и checkpoint/resume.
5. **[готово для shadow] Dialogue foundation.** History, typed pending state,
   generic clarification, reminder slot merge, state logging и reset policy.
6. **[готово для shadow] Semantic safety layer.** Grounding каждого шага,
   targetless-close blocker, non-execute draft isolation, negation/process
   blockers и canonical application allowlist.
7. **[готово] Data-first v8 cycle.** 4 355 train-примеров,
   category-balanced sampling, seed comparison и migration gates.
8. **[готово] Experimental production wiring.** Seed29 загружается из
   `config.yaml` в `jsc_shadow` с validation-selected thresholds.

## Выбранный checkpoint

- Release path: `models/jsc/structured_v8_seed29.pt`.
- SHA-256: `968ff79119fb7fc46b0023c813025fc28a9f755451807b8cb49726441cadb5ec`.
- Parameters: 534 942.
- Topology: `d_model=96`, 2 encoder layers, FFN 192.
- Best epoch: 5.
- Thresholds: execution 0,65; verifier 0,90; parameter 0,35; span 0,25;
  missing 0,35.

## Последний benchmark

Migration development, 400 примеров:

| Метрика | Результат |
|---|---:|
| Exact JAL | 87,75% |
| Act | 94,00% |
| Tool sequence | 95,75% |
| Arguments | 93,75% |
| Single | 90,00% |
| 2–3 actions | 93,02% |
| 4–5 actions | 100,00% |
| Multi-turn | 100,00% |
| ASR noise | 100,00% |
| Hard negative | 83,33% |
| Correction | 46,67% |
| OOD exact | 33,33% |
| Schema valid | 100,00% |
| False execution | 0,00% |
| Opposite action | 0,00% |

CPU production-wrapper smoke: warm примерно 6–129 мс, первый cold request
около 0,5–0,8 с.

Migration development не является новым frozen holdout: результаты
использовались при анализе decoder/data. Цифра 87,75% разрешает shadow wiring,
но не доказывает качество на любой пользовательской команде.

## Активный этап — correction, OOD и independent voice holdout

### P0. Correction transaction

- хранить target предыдущего действия и его execution outcome;
- различать новую команду, replacement и отмену;
- определить compensation policy для уже открытого/закрытого приложения;
- журналировать исходный plan, correction plan, compensation и final state;
- довести correction exact с 46,67% до promotion gate.

### P0. OOD и selective risk

- отделить unsupported tool, свободный диалог и недостаток данных;
- обучить/калибровать OOD signal без ослабления execution/verifier thresholds;
- измерять selective precision на auto-execute subset;
- сохранить false execution 0% и opposite action 0%.

### P0. Frozen voice holdout

- записать новый набор после фиксации data/decoder/thresholds;
- не использовать его в обучении, alias rules и debugging;
- включить реальные одиночные, 2–5 actions, multi-turn, corrections, rejects,
  OOD, ASR noise и long-tail applications;
- оценивать semantic plan и фактический end-to-end outcome отдельно;
- хранить private audio только по явному consent и вне Git.

### P1. Runtime telemetry

- накопить новый `logs/jsc_shadow.jsonl` именно с seed29;
- измерить CPU p50/p95, cold start, memory и long-session drift;
- сравнить production NLU/JSC на одинаковых traces;
- добавить offline replay без side effects;
- зафиксировать disagreement classes и приоритет следующего data cycle.

## Promotion gates

Минимальные gates перед design review execution canary:

- новый frozen voice Exact JAL >= 0,90;
- multi-turn end-to-end >= 0,95;
- correction >= 0,90;
- OOD recall >= 0,97;
- auto-execute precision >= 0,99;
- false execution <= 0,1%, целевое значение 0%;
- opposite action = 0%;
- schema-valid = 100%;
- CPU warm p95 <= 100 мс на целевой машине;
- ноль draft/targetless/destructive hallucinations;
- отсутствие регрессии production tool families.

Execution canary сначала должен быть no-side-effect replay. Возможное реальное
выполнение проектируется отдельным решением после review. Удаление NLU — ещё
более поздний этап и не следует автоматически из успешного canary.

## Дальнейшие этапы

1. **Reliability.** Independent plan verifier, calibrated abstention и
   disagreement policy.
2. **Tool-schema scaling.** Новые tools только вместе с data, grounding,
   regression и holdout.
3. **Personalization.** Только явный feedback/opt-in adapters; никакого скрытого
   обучения на личных логах.
4. **Final multi-seed sweep.** Accuracy/latency Pareto после стабилизации
   correction/OOD architecture, а не до неё.
5. **Promotion.** Shadow -> offline canary -> reviewed restricted canary ->
   production decision -> отдельное решение об удалении NLU.

## Решение на текущую дату

Structured JSC v8 seed29 — выбранный experimental production shadow checkpoint.
Он достиг пользовательского промежуточного диапазона 85–90% на migration
development и безопасен в текущем shadow-контуре. Production promotion пока
запрещён из-за correction/OOD и отсутствия нового frozen voice holdout.
