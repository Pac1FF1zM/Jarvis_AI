# Jarvis — модульный локальный голосовой ассистент

Jarvis — событийный Python-ассистент для Windows. Модули не вызывают друг
друга напрямую: данные проходят через `EventBus`, а `Orchestrator` контролирует
состояния `IDLE → LISTENING → TRANSCRIBING → THINKING → TOOL_CALL → SPEAKING`.

## Что работает сейчас

`python main.py` запускает постоянный push-to-talk режим. Нажатие глобальной
комбинации `Ctrl+Alt+Space` начинает запись с микрофона; Silero VAD завершает
её после окончания речи, после чего запускается полный цикл:

```text
wake_word_detected
  → audio_captured
  → transcription_ready
  → nlu_result
  → tool_call_requested / general chat
  → response_ready
  → speech_started
  → speech_finished
  → IDLE
```

- Активация пока выполняется горячей клавишей, а не фразой «Джарвис».
  Захват микрофона настоящий: `sounddevice`, mono PCM16, 16 кГц, с Silero VAD
  и ограничением максимальной длительности записи. Без аудиобиблиотек доступен
  безопасный одноразовый режим `python main.py --demo`.
- STT подключён к официальному `openai-whisper` без Hugging Face; без пакета
  или настоящего аудио использует
  безопасную заглушку.
- NLU — собственная нейросеть проекта, обученная с нуля. Она выбирает
  намерение и маршрутизирует инструменты.
- Ollama используется только для свободного диалога. Решения о запуске
  инструментов принимает NLU, а готовый результат возвращается напрямую;
  при недоступности Ollama работает текстовая заглушка.
- TTS настроен на русский Silero `v4_ru` (`xenia`, 48 кГц) и `sounddevice`;
  без них работает заглушка.
- Доступны инструменты текущего времени, списка приложений и безопасного запуска
  приложений. Запрос напоминания честно отклоняется: постоянного планировщика нет.

## Собственная ML-часть

В `ml/nlu/` находятся собственные:

- корпус русских команд с независимыми train/validation/test-шаблонами;
- собственные символьный и словный токенизаторы;
- символьные `CharCNN`/`BiGRU` и словная `Word-BiGRU` для intent
  classification и BIO slot tagging;
- режимы обучения `standard`, `augmented`, `curriculum`;
- обучение, оценка, checkpoints и runtime inference.

Hugging Face, готовые веса, готовые embeddings, внешние токенизаторы и
скачанные датасеты не используются. PyTorch служит только вычислительным
фреймворком.

Выбранный runtime checkpoint — `models/nlu_word_bigru_curriculum.pt`:

| Модель и обучение | Validation intent macro-F1 | Test intent macro-F1 | Test raw slot F1 |
|---|---:|---:|---:|
| Word-BiGRU + standard | 0.802 | 0.803 | 0.678 |
| Word-BiGRU + curriculum | **0.908** | **0.818** | **0.670** |
| CharCNN + augmented | 0.662 | 0.807 | 0.527 |

Raw neural slot tagging ещё недостаточно точен. Первый runtime baseline поэтому
честно является гибридным: нейросеть принимает решение о намерении, а два
параметра текущего reminder-инструмента проверяются ограниченным декодером. На
development-test шаблонах полный набор параметров извлечён в 27/27 случаев.
Отдельный замороженный holdout из 49 новых фраз дал 0.816 accuracy и 0.818
macro-F1; его ошибки не возвращались в обучение. Подробности:
[`ml/README.md`](ml/README.md).

Запуск приложений дополнительно требует явной командной конструкции и проходит
через белый список: Калькулятор, Блокнот, Проводник, Paint, Диспетчер задач и
системный браузер. Произвольные команды shell модель выполнять не может.

## Установка и запуск

Рекомендуется Python 3.10+ на Windows:

```powershell
cd "C:\Users\Hp Victus\Desktop\Jarvis"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py               # постоянный режим, Ctrl+Alt+Space для записи
python main.py --demo        # один цикл без микрофона и глобальной клавиши
```

Команду также можно передать текстом, полностью минуя микрофон и STT:

```powershell
python main.py --text "открой калькулятор"
python main.py --text "какие приложения ты можешь открыть"
python main.py --text "сколько сейчас времени"
```

Текст проходит через ту же собственную NLU-модель, оркестратор и инструменты;
пропускаются только симулированные захват аудио и STT.

Каждый запуск автоматически сохраняет отдельный UTF-8 лог в
`logs/sessions/jarvis_session_*.txt`. Путь печатается при старте как
`SESSION_LOG_READY`. Файл содержит распознанный Whisper текст, confidence,
решение NLU, параметры инструмента, ответ и ошибки; его можно прикладывать как
feedback для подготовки следующего обучающего корпуса. `logs/jarvis.log`
по-прежнему содержит только последний запуск для анализа задержек.

Повторное обучение выбранного baseline:

```powershell
python -m ml.nlu.train --architecture word_bigru --method curriculum `
  --epochs 40 --batch-size 64 --hidden-dim 64 --max-length 32 `
  --output models/nlu_word_bigru_curriculum.pt
```

Сравнительные эксперименты:

```powershell
python -m ml.nlu.train --architecture word_bigru --method standard `
  --epochs 30 --batch-size 64 --hidden-dim 64 --max-length 32 `
  --output models/nlu_word_bigru_standard.pt
python -m ml.nlu.train --architecture char_cnn --method augmented `
  --epochs 25 --output models/nlu_cnn_augmented.pt
```

Каждый checkpoint содержит случайный seed, vocabulary, конфигурацию, веса и
метрики. Рядом сохраняется читаемый `*.metrics.json`.

Проверка на замороженном holdout:

```powershell
python -m ml.nlu.evaluate_holdout --checkpoint models/nlu_word_bigru_curriculum.pt
```

## Fine-tuning на отдельном GPU-компьютере

Готовое переносимое пространство находится в [`training_workspace/`](training_workspace/).
Оно продолжает обучение собственного checkpoint без Hugging Face, расширяет
локальный vocabulary, сравнивает три режима обучения, измеряет accuracy/latency
и экспортирует только кандидата, который не хуже baseline.

На компьютере с RTX 3090:

```powershell
.\training_workspace\SETUP_RTX3090.ps1
.\training_workspace\START_TRAINING.ps1 -CheckOnly
.\training_workspace\START_TRAINING.ps1
```

Формат данных, правила независимого holdout, интерпретация отчёта и безопасное
подключение победителя описаны в
[`training_workspace/GUIDE_RU.md`](training_workspace/GUIDE_RU.md).

## Проверка

```powershell
python -m pytest -v
```

Тесты включают полный событийный цикл, несколько последовательных push-to-talk
циклов, мокированный микрофон и VAD, настоящий NLU checkpoint, сохранение
`trace_id`, выполнение NLU/STT/TTS вне event-loop потока и конкурентную отмену
TTS.

## Структура

```text
core/       EventBus, Orchestrator, GPULock, конфигурация
modules/    wake word, STT, собственный NLU, LLM, TTS
ml/nlu/     датасет, токенизатор, модели, обучение, inference
models/     checkpoints и метрики экспериментов
tools/      автоматически обнаруживаемые инструменты
memory/     краткосрочная память и SQLite-хранилище
tests/      unit, regression и end-to-end тесты
```

Конфигурация модулей находится в `config.yaml`. На GTX 1060 3 ГБ NLU работает
на CPU; GPU-доступ остальных моделей сериализуется через `GPULock`.

## Пока не реализовано

- постоянное прослушивание и настоящее распознавание фразы активации;
- подтверждённая на реальном микрофоне/колонках работа всей голосовой цепочки;
- реальное срабатывание запланированного напоминания;
- закрытие приложений и управление их интерфейсом после запуска;
- использование долгосрочной памяти в диалоге;
- полностью neural exact slot extraction без constrained decoder;
- проверка всей цепочки barge-in на реальном микрофоне и аудиоустройстве.
