# План миграции NLU → JSC/JAL

## Текущее состояние — Stage 1: agreement canary

Release 0.9.0 включает всю управляемую цепочку миграции. Structured JSC и
legacy NLU получают один transcript независимо, а `JSCMigrationModule` является
единственным владельцем semantic routing. В активной стадии результат NLU
уходит дальше как `semantic_result`, JSC не исполняет команды, а канонические
NLU/JSC JAL-планы сравниваются в `logs/jsc_agreement.jsonl`.

Стадия задаётся в `config.yaml`, но одного изменения конфигурации недостаточно:
runtime читает versioned evidence `models/JSC_MIGRATION_STATE.json` и при
несоблюдении ворот автоматически возвращается в `agreement_canary`. Текущие
значения — 0 reviewed voice turns и 0 stable release cycles; офлайн-прогон на
400 примерах зафиксирован отдельно и не считается human voice evidence.

## Stage 2: restricted reversible

Код готов, promotion пока заблокирован. После прохода human voice gates JSC
получит primary routing только для schema-valid, complete и calibrated accepted
планов из обратимого allowlist. Все остальные запросы останутся на NLU.

`JALExecutorModule` уже реализует:

- последовательную JAL-транзакцию под одним trace;
- обязательную compensation evidence для изменяющих состояние canary-действий;
- rollback уже выполненных шагов при ошибке следующего шага;
- correction transaction `compensation → verify → replacement`;
- запрет replacement при ошибке compensation;
- восстановление исходного действия при ошибке replacement;
- committed-action receipt для следующего correction turn.

Promotion требует не менее 1 000 размеченных голосовых turns минимум от трёх
пользователей: false execution = 0, opposite action = 0, semantic exact ≥ 90%,
correction ≥ 95%, OOD recall ≥ 98%. Rolling error budget немедленно отключает
promoted routing при unsafe disagreement и возвращает agreement canary.

## Stage 3: JSC/JAL primary

Код готов, promotion заблокирован минимум до одного стабильного release-цикла
после прохода voice gates. JSC становится semantic owner для поддержанных JAL
планов; incomplete, OOD и high-risk запросы fail closed через calibrated
abstention/reject. NLU остаётся включённым shadow-контролем и rollback-путём.

## Stage 4: NLU removed

Условный импорт NLU уже позволяет собрать runtime без legacy-модуля и
checkpoint. Фактическое удаление из runtime, installer и release manifest
разрешено только после двух последовательных стабильных JSC-primary
release-циклов. На один следующий релиз NLU checkpoint сохраняется как
отдельный rollback artifact.

## Незакрытый внешний gate

В репозитории есть независимый paired FLEURS STT benchmark: Parakeet WER 4,40%
против Whisper-small 5,35%. Это общая русская речь, а не команды Jarvis.
Production promotion требует свежий приватный human-command manifest и
размеченную seed29 telemetry. Runner и consent-aware формат описаны в
`benchmarks/voice_e2e/README_RU.md`; readiness проверяется командой:

```powershell
.venv-training\Scripts\python.exe training_workspace\check_jsc_migration_readiness.py
```

Текущий ожидаемый результат: agreement canary admitted, все исполняющие стадии
blocked. Это защитный статус, а не незавершённая реализация runtime-путей.
