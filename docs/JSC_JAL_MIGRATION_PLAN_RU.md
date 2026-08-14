# План миграции NLU → JSC/JAL

## Текущее состояние — Stage 0: independent shadow

Structured JSC получает `transcription_ready` напрямую, до результата старой
NLU. Он публикует только `jsc_candidate_ready` с неизменяемым
`execution_allowed: false`, пишет risk/completeness/transaction telemetry и не
имеет пути к инструментам. NLU остаётся production router и контрольной
группой.

## Stage 1: agreement canary

Собрать не менее 1 000 свежих голосовых turns минимум от трёх пользователей.
Разметить disagreement, correction, OOD, incomplete/compound и ASR-noise.
Обязательные ворота: false execution = 0, opposite action = 0, schema validity
= 100%, correction ≥ 95%, OOD recall ≥ 98%, full voice semantic exact ≥ 90%.
Canary остаётся без исполнения.

## Stage 2: restricted JSC primary

Разрешить JSC только обратимые allow-listed действия при успешных calibrated
risk и completeness gates. Correction выполняется одной транзакцией:
compensation → проверка результата → replacement; при ошибке компенсации
replacement не запускается. NLU работает в shadow и автоматически откатывает
stage при превышении error budget.

## Stage 3: JSC/JAL primary

После двух стабильных release-циклов перевести все поддержанные инструменты на
JAL executor, удалить NLU из runtime и installer, но сохранить отдельный
rollback artifact на один релиз. Неизвестные и высокорисковые запросы всегда
abstain/reject; расширение grammar требует новой frozen benchmark версии.

## Незакрытый внешний gate

В репозитории есть независимый paired FLEURS STT benchmark (Parakeet WER 4,40%
против Whisper-small 5,35%), но это общая русская речь, не набор команд Jarvis.
Для production-перехода всё ещё нужен приватный human-command manifest: не
менее 30 записей, трёх спикеров и матрица чистая речь/шум/дальний микрофон.
Runner и формат описаны в `benchmarks/voice_e2e/README_RU.md`.
