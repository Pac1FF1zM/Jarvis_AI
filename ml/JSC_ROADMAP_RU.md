# Jarvis Semantic Core — рабочий roadmap

Цель: заменить фиксированный intent-классификатор собственной нейросетевой
системой, которая понимает контекст, строит исполняемый план, обнаруживает
ошибки и выбирает `execute / ask / confirm / cancel / reject`. Все веса,
токенизатор и обучающие генераторы принадлежат проекту; Hugging Face, готовые
weights/embeddings и внешние AI API не используются.

## Этапы

1. **[готово] JAL v1 и контракт инструментов** — формальный язык планов, канонический
   codec, типизированная проверка по реальным tool schemas и тесты.
2. **[готово] Фабрика данных v5** — воспроизводимые одиночные, составные и многотуровые
   сценарии; ASR-искажения, hard negatives, OOD, цифры и русские числительные;
   независимые family splits.
3. **[готово] Честные baselines** — CharCNN, BiGRU и
   tiny Transformer encoder с общим JAL decoder; одинаковые данные, три seed,
   checkpoint/resume и единый safety-oriented evaluation protocol.
4. **[в работе] JSC semantic parser** — собственный Transformer encoder-decoder,
   copy-механизм, числовые признаки, отдельные головы типа действия, числа
   шагов, упорядоченного списка инструментов, категориальных параметров и
   свободных character spans; schema-conditioned сборщик и fail-closed выдача
   JAL. Следующий шаг — улучшение обобщения act/tool/span heads на новых
   семействах формулировок и независимый verifier.
5. **Диалог и исправления** — neural dialogue-state tracker, пропущенные slots,
   отрицания, замены, подтверждения и составные команды.
6. **Надёжность** — OOD head, conformal clarification, selective risk,
   независимый plan verifier и запрет уверенного исполнения при disagreement.
7. **Персонализация** — явный feedback, user adapters и replay без
   catastrophic forgetting; никакого скрытого обучения на личных данных.
8. **Финальный sweep и runtime** — multi-seed/ablation, Pareto accuracy-latency,
   frozen voice holdout, approved checkpoint и интеграция с orchestrator.

## Главные гейты

- exact JAL program accuracy ≥ 0.92 на реальных in-domain командах;
- успешность многотурового сценария ≥ 0.95;
- OOD recall ≥ 0.97 и ложный запуск инструмента ≤ 0.1%;
- selective precision автоматически исполненных планов ≥ 0.99;
- 100% выданных планов проходят grammar/schema validation;
- CPU p95 ≤ 50 мс и отсутствие регрессии старых навыков;
- holdout не участвует в подборе параметров или эпохи.

Текущий этап отмечается в git и меняется только после тестов и измеримого
выходного гейта. Baseline-протокол из 21 запуска выполнен на RTX 3050 6GB.
Лучший validation-эксперимент этапа 4 на сбалансированном JSC v5 (Transformer,
seed 17, execution threshold 0,90) использует categorical parameter head,
character-span head, отдельный execution verifier и полную детерминированную
грамматику только как второй сигнал при согласии verifier. Он дал 100%
schema-valid планов, 0% ложных исполнений, 28,97% exact JAL, 37,59% argument
sequence accuracy, 100% OOD recall и 77,50% точности значения для чисел
словами. Это +4,14 п.п. exact JAL относительно v7. Переобучившийся вариант с
новым semantic pooling отклонён; encoder и старые головы в выбранном v8
остались заморожены.

На один раз открытом после выбора v7 test было: 100% schema validity, 18,62%
exact JAL, 27,24% argument accuracy, 76% OOD recall и 0,34% ложных исполнений.
V8 подбирался только на validation и повторно на test не проверялся. Эти
результаты всё ещё запрещают promotion исследовательского checkpoint
`training_workspace/jsc_runs_v8/legacy_verifier_seed17/best.pt` в runtime.
Контрольный BiGRU дал 18,62% exact JAL на validation. Ключевая
четырёхшаговая фраза без пунктуации корректно собрана в
`open_application → gesture_mode → set_reminder → window_control`, включая
число «двадцать». До production-гейтов стабильная NLU не заменяется.
Подробности:
`ml/jsc/JAL_SPEC_RU.md`,
`training_workspace/jsc_data/README_RU.md` и
`training_workspace/JSC_BASELINES_RU.md`.
