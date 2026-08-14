# Jarvis — полный статус проекта, актуализирован 15 августа 2026

Этот документ — актуальная точка входа для продолжения разработки. Старые
checkpoint-документы сохраняются как история и не являются источником текущей
production-конфигурации.

## Краткий итог

Jarvis 0.9.0 работает как локальный Windows-ассистент с production Parakeet
STT, безопасными инструментами, Silero TTS, памятью, напоминаниями, Control
Center и ограниченным Gesture Core. Миграция на Structured JSC/JAL переведена
в активный `agreement_canary`: JSC и legacy NLU анализируют один transcript
независимо, но только единый coordinator выбирает semantic owner. В canary им
остаётся NLU; JSC не имеет side effects.

Restricted reversible, JSC-primary и NLU-removed runtime-пути реализованы и
покрыты тестами. Они fail-closed заблокированы versioned evidence до реальных
human voice gates и одного/двух стабильных release-циклов. Это означает, что
архитектура готова к контролируемой миграции, но production promotion ещё не
получил достаточных полевых доказательств.

Текущие уровни зрелости:

| Контур | Статус | Разрешённые side effects |
|---|---|---|
| Основной голосовой runtime | Production/local, agreement canary | Да, только выбранный NLU control |
| Parakeet diagnostic | No-action diagnostic | Нет |
| Structured JSC v8 | Active agreement canary | Нет |
| Restricted JAL executor | Code-ready, evidence-blocked | Нет до promotion |
| JSC-primary / NLU removed | Code-ready, release-cycle-blocked | Нет до promotion |
| Gesture Core | Restricted real-camera test | Только G01–G06 |
| Свободный диалог Ollama | Optional/degraded в последней сессии | Инструменты не маршрутизирует |

## Runtime pipeline

```text
openWakeWord / Ctrl+Alt+Space
  -> Silero VAD + PCM16 16 kHz capture
  -> Parakeet TDT 0.6B v3
  -> semantic commit gate
  -> production NLU + Structured JSC v8 (independent)
  -> migration coordinator
     -> semantic_result from NLU (active agreement canary)
     -> JAL transaction request (only after admitted promotion)
  -> orchestrator + schema-checked tools / JAL executor
  -> response
  -> Silero TTS
```

Основной запуск — `START_JARVIS_UI.cmd` / Jarvis Control Center. Прямой
developer-запуск — `venv\Scripts\python.exe main.py`. `main.py --demo` остаётся
безопасным текстовым smoke, а `main.py --gesture_mode` запускает отдельный
gesture runtime.

## Активные модули

| Модуль | Реальная конфигурация | Текущее состояние |
|---|---|---|
| Wake word | openWakeWord `hey_jarvis`, threshold 0,35; PTT `Ctrl+Alt+Space` | Включён; active session 7 с |
| STT | `nvidia/parakeet-tdt-0.6b-v3`, локальный persistent worker, CUDA auto | Единственный production STT |
| Legacy NLU control | `models/nlu_manager_finetuned.pt`, CPU, threshold 0,55 | Активный owner/fallback в agreement canary |
| Structured JSC | Structured v8 seed29, CPU, state/history | Agreement candidate, execution отсутствует |
| Migration coordinator | Evidence admission + rolling error budget | Единственный semantic owner |
| JAL executor | Schema, reversibility, compensation, rollback | Готов; не активирован evidence-gate |
| LLM | Ollama `qwen2.5:7b-instruct`, CPU Q4_K_M | Опционален; в последних сессиях недоступен, используется stub |
| TTS | Silero `v4_ru`, `xenia`, 48 kHz | Включён, prewarm + bounded cache |
| Gesture | Jester Tiny3D, CUDA auto | Restricted test; allowlist G01–G06 |
| Memory | SQLite + профильные факты | Включена; обычный диалоговый контекст ограничен |
| Reminders | SQLite scheduler, poll 0,5 с | Создание, список, отмена и срабатывание |
| Feedback | локальная очередь uncertain/tool failures | Включён; автоматического обучения нет |

## Модели и артефакты

### Parakeet production STT

- Snapshot: `.local/parakeet/models/nvidia--parakeet-tdt-0.6b-v3`.
- Локальный размер snapshot: около 2,51 ГБ.
- Whisper удалён из runtime, config, installer dependency и Doctor; остался
  только исторический сравнительный benchmark.
- FLEURS `ru_ru`, 20 общих human-speech клипов <=12 с:
  Parakeet WER 4,40%, exact 70%, mean decode 1210 мс; Whisper small WER 5,35%,
  exact 50%, mean decode 2149 мс.
- Это публичный smoke, а не закрытый benchmark боевых команд.

### Production NLU

- Checkpoint: `models/nlu_manager_finetuned.pt`.
- SHA-256: `d30d2ba11398124b3b721fe2aae56dd715b36ce15c9aaa08296408e3beacec7b`.
- CharCNN + augmented, 169 199 параметров.
- Frozen holdout intent macro-F1 0,959; constrained slot decoders и semantic
  gate остаются частью production-системы.
- NLU загружается условно и технически больше не является обязательным импортом.
  В 0.9.0 он намеренно остаётся control/fallback; удаление из пакета разрешено
  только после двух стабильных JSC-primary release-циклов.

### Structured JSC v8

- Release checkpoint: `models/jsc/structured_v8_seed29.pt`.
- SHA-256: `968ff79119fb7fc46b0023c813025fc28a9f755451807b8cb49726441cadb5ec`.
- 534 942 параметра; `d_model=96`, 2 encoder layers, best epoch 5.
- Train: 4 355 schema-valid примеров с category-balanced sampling.
- Прямые structured heads; autoregressive JSON generation отсутствует.
- Validation-selected thresholds: execution 0,65; verifier 0,90; parameter
  0,35; span 0,25; missing 0,35.

Migration development:

| Метрика | Результат |
|---|---:|
| Exact JAL | 87,75% |
| Act accuracy | 94,00% |
| Tool sequence | 95,75% |
| Argument sequence | 93,75% |
| Single | 90,00% |
| 2–3 действия | 93,02% |
| 4–5 действий | 100,00% |
| Multi-turn | 100,00% |
| ASR noise | 100,00% |
| Hard negative | 83,33% |
| Correction | 46,67% |
| OOD exact | 33,33% |
| Schema valid | 100,00% |
| False execution | 0,00% |
| Opposite action | 0,00% |

Свежий versioned offline runtime gate на 400 примерах: Exact JAL 96,75%,
correction 100%, OOD recall 100%, false execution 0%, opposite action 0%.
Отдельный seed29 probe: 24/24. Эти результаты подтверждают код и frozen
артефакт, но не засчитываются как новые voice turns.

CPU-smoke после прогрева: примерно 6–129 мс; первый cold request около
0,5–0,8 с. JSC задаёт typed clarification для неизвестного приложения/окна и
неполного reminder, затем заполняет pending slot следующим ходом.

Ограничение результата: migration development и offline runtime gate не
являются новым frozen voice holdout. `models/JSC_MIGRATION_STATE.json` честно
фиксирует `reviewed_voice_turns: 0` и `stable_release_cycles: 0`; свежая полевая
seed29 telemetry ещё не накоплена.

### Gesture Core

- Release checkpoint: `models/gesture/20260812_jester_tiny3d/best.pt`.
- SHA-256: `0c14820042c652c8b51d19c82af57fd0f0706edcf317f1859cf2e461b80f7e07`.
- Official test: 14 743 клипа; accuracy 0,7922; macro-F1 0,7871;
  negative recall 0,8962.
- В side effects переводятся только G01–G06; остальные 21 класса становятся
  D0X. Это ограниченная real-camera проверка, не финальная production-модель.

## Реализованные возможности

- запуск allow-listed и безопасно обнаруженных Windows-приложений;
- управление окнами, включая составные команды закрытия/открытия;
- браузер: поиск, сайты и вкладки;
- пользовательские файлы: поиск, список, открытие, reveal, создание папки,
  переименование и удаление через корзину;
- системные настройки, громкость, media controls и блокировка ПК;
- reminders: relative/absolute time, list, cancel by id;
- рабочие пространства;
- undo для поддерживаемых обратимых действий;
- typed confirmation для опасных операций;
- compound plan на 2–5 действий с fail-closed сборкой;
- clarification и state carry-over для приложения, окна и reminder;
- долгосрочная память, профили и микрофонная калибровка;
- Control Center: запуск/остановка, Doctor, настройки, калибровка, логи и
  встроенный gesture preview;
- локальная session telemetry без автоматического добавления личных данных в
  train-набор.

Автоматически обнаруживаемые tools в текущем registry:
`browser_control`, `cancel_reminder`, `file_control`, `get_current_time`,
`list_applications`, `list_reminders`, `open_application`, `set_reminder`,
`system_control`, `undo_action`, `window_control`, `workspace_control`.

## Voice feedback A–G

Старый production/JSC цикл показал ошибки compound routing, targetless close,
ASR-алиасов, reminder slot merge и отсутствие уточнений. После него:

- JSC shadow получил history, typed pending state и `dialogue_id`;
- generic app/window и incomplete reminder возвращают `ask`;
- non-execute draft не может попасть в executor;
- добавлены semantic grounding и blockers для negation/process-level команд;
- добавлены ASR-варианты `вэ скот`, `вскод`, `дискод`, `паинт`, `телегу`,
  `отпрой` и trailing wake word;
- natural compound data расширены до 2–5 действий;
- seed29 поднял migration exact до 87,75%.

Старые A–G логи нельзя использовать как независимое подтверждение новой
модели: их ошибки повлияли на данные и decoder. Нужен новый закрытый прогон.

## Последний runtime health audit

Последняя длинная voice-сессия: 14 августа 2026, 14:41–15:18. Критических
ошибок в пяти последних session logs не найдено. Наблюдались:

- Ollama недоступен на startup probe — свободный диалог использовал быстрый
  stub до перезапуска;
- один launch warning: Discord показал splash, но основное окно не стало
  доступно в отведённое время;
- `MICROPHONE_EMPTY` после 7 секунд active session — штатный возврат ко сну;
- Gesture Core пишет warning о restricted allowlist и готовности preview — это
  текущий тестовый режим, а не авария.

## Проверки

- Новый migration runtime suite проверяет stage admission, обязательные два
  release-цикла, unsafe error budget, canonical legacy adapter, agreement
  forwarding, restricted selection, compensation и compound rollback.
- Runtime readiness CLI подтверждает: agreement canary admitted; restricted,
  JSC-primary и NLU-removed корректно blocked текущим human evidence.
- Versioned snapshot результата сохранён в
  `docs/evidence/JSC_MIGRATION_READINESS_20260814.json`.
- Полный suite в `.venv-training`: **570 passed, 3 skipped** за 1:44;
  ошибок нет. Восемь предупреждений относятся к известному PyTorch
  `TransformerEncoder nested_tensor/norm_first` и не меняют результат.
- `git diff --check`: ошибок whitespace нет; предупреждения LF/CRLF ожидаемы на
  Windows.
- Checkpoint SHA-256 и пути из `config.yaml` подтверждены локально.

Runtime `venv` намеренно не содержит `tensorboard`, поэтому общий pytest в нём
не собирает training-only IPN/Jester tests. Полный общий suite следует запускать
из `.venv-training`; runtime-only suite — с исключением двух training files, как
описано в README.

## Git и воспроизводимость

- Production-код, migration coordinator/JAL executor, versioned evidence,
  Structured JSC v8 release export, Jester Tiny3D release export и их audit
  reports входят в `main` и Windows Setup.
- `config.yaml` использует только стабильные release-пути внутри `models/`;
  clone больше не зависит от локальных training run directories.
- Тяжёлые training snapshots, private logs, WAV, databases и generated reports
  не попадают в Git. Личные логи не становятся train data автоматически.

## Незакрытые риски

1. Нет нового frozen voice holdout после fine-tuning; offline 96,75% нельзя
   обобщать на любую пользовательскую фразу.
2. Нет живой agreement telemetry seed29 после обновления `config.yaml`.
3. NLU и JSC ещё не сравнивались на одном новом закрытом голосовом
   наборе после исправлений.
4. Human-gates correction ≥95% и OOD recall ≥98% ещё не подтверждены, несмотря
   на 100% в offline runtime gate.
5. Stable JSC-primary release cycles: 0 из 2; NLU удалять нельзя.
6. Ollama в последних сессиях выключен; свободный диалог работает через stub.
7. Discord cold launch иногда не подтверждает появление основного окна.
8. Gesture Core требует персональной camera calibration и более сильного
   real-camera acceptance перед расширением allowlist.
9. Installer/Control Center ещё не оформлены как подписанный конечный `.exe`.

## Следующие действия по приоритету

### P0 — пройти agreement canary evidence

1. Накопить не менее 1 000 свежих размеченных `jsc_agreement` voice turns
   минимум от трёх пользователей: single, compound, clarification, correction,
   reject, OOD, шум и дальний микрофон.
2. Заморозить private human-command holdout до любых следующих изменений
   decoder/data/thresholds и прогнать audio → Parakeet → semantic plan E2E.
3. Подтвердить false execution/opposite action 0%, semantic exact ≥90%,
   correction ≥95% и OOD recall ≥98%; измерить CPU p50/p95/cold start.
4. Записать проверенные метрики в `models/JSC_MIGRATION_STATE.json` и только
   после review переключить stage на `restricted_reversible`.
5. Провести один стабильный restricted/JSC-primary цикл с error budget, затем
   второй стабильный JSC-primary цикл. Только после них выставлять
   `nlu_removed` и исключать NLU из installer/manifest.

### P1 — runtime polish

1. Включить/закрепить Ollama через Control Center или явно оставить stub режим.
2. Улучшить подтверждение Discord cold start.
3. Завершить installer entrypoint, упаковку и подпись.
4. Провести end-to-end barge-in acceptance на реальном микрофоне/выводе.

### P2 — дальнейшее развитие

1. Персональная opt-in gesture calibration без сохранения видео по умолчанию.
2. Opt-in speaker verification как отдельная биометрическая функция.
3. Расширение tool schemas и training data только вместе с regression/holdout.

## Решение на текущую дату

Parakeet остаётся единственным production STT. Release 0.9.0 находится на
Stage 1 `agreement_canary`: NLU — активный semantic control, JSC — независимый
неисполняемый кандидат. Вся цепочка restricted JAL → JSC-primary → условное
удаление NLU реализована, протестирована и fail-closed управляется evidence.
Её фактическое включение отложено до настоящей seed29 voice telemetry и двух
стабильных циклов; синтетические результаты за них не выдаются. Gesture Core
остаётся restricted test.
