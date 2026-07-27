# Jarvis Semantic Core — рабочий roadmap

Цель: заменить фиксированный intent-классификатор собственной нейросетевой
системой, которая понимает контекст, строит исполняемый план, обнаруживает
ошибки и выбирает `execute / ask / confirm / cancel / reject`. Все веса,
токенизатор и обучающие генераторы принадлежат проекту; Hugging Face, готовые
weights/embeddings и внешние AI API не используются.

## Этапы

1. **JAL v1 и контракт инструментов** — формальный язык планов, канонический
   codec, типизированная проверка по реальным tool schemas и тесты.
2. **Фабрика данных v3** — воспроизводимые одиночные, составные и многотуровые
   сценарии; ASR-искажения, hard negatives, OOD; независимые family splits.
3. **Честные baselines** — текущий CharCNN, BiGRU-CRF и DIET-подобный encoder;
   одинаковые данные, seeds и evaluation protocol.
4. **JSC semantic parser** — собственный Transformer encoder-decoder,
   grammar-constrained генерация JAL и schema-conditioned tool selection.
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
выходного гейта. Подробная спецификация первого этапа: `ml/jsc/JAL_SPEC_RU.md`.
