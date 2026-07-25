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
- STT подключён к многоязычному `small` из официального `openai-whisper` без
  Hugging Face. При доступной NVIDIA CUDA он работает на GPU в FP16, иначе
  переходит на CPU/FP32; без пакета или настоящего аудио использует безопасную
  заглушку.
- NLU — собственная нейросеть проекта, обученная с нуля. Она выбирает
  намерение и маршрутизирует инструменты.
- Ollama используется только для свободного диалога. Решения о запуске
  инструментов принимает NLU, а готовый результат возвращается напрямую;
  при недоступности Ollama работает текстовая заглушка.
- TTS настроен на русский Silero `v4_ru` (`xenia`, 48 кГц) и `sounddevice`;
  без них работает заглушка.
- Необработанное исключение handler превращается в `interaction_failed`;
  watchdog возвращает зависший trace в `IDLE`, а запоздавшие события закрытого
  trace игнорируются. Recovery также останавливает захват и TTS.
- Доступны инструменты текущего времени, списка приложений и безопасного запуска
  приложений. Напоминания сохраняются в SQLite, переживают перезапуск Jarvis,
  срабатывают голосом в постоянном режиме, а также поддерживают просмотр и
  отмену по номеру. Созревшее напоминание ждёт завершения активного диалога.

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

Выбранный runtime checkpoint — `models/nlu_manager_finetuned.pt`. Это
`CharCNN + augmented` с 53 919 параметрами:

| Проверка | Intent macro-F1 | Худший recall intent |
|---|---:|---:|
| Custom validation (210 фраз) | 0.919 | 0.800 |
| Legacy regression | 0.964 | 0.750 |
| Evaluation holdout (105 фраз) | 0.839 | 0.667 |
| Замороженный holdout (49 фраз) | **0.959** | **0.857** |

Raw neural slot tagging ещё недостаточно точен. Первый runtime baseline поэтому
честно является гибридным: нейросеть принимает основные решения о намерении,
а параметры запуска приложений и напоминаний проверяются ограниченными
декодерами. Команды создания, просмотра и отмены напоминаний также имеют узкие
детерминированные правила, не подменяющие свободный диалог.
Ни один holdout не используется для обновления весов или выбора эпохи.
Подробности:
[`ml/README.md`](ml/README.md).

Запуск приложений дополнительно требует явной командной конструкции и проходит
через белый список: Калькулятор, Блокнот, Проводник, Paint, Диспетчер задач и
системный браузер. Произвольные команды shell модель выполнять не может.

## Установка и запуск

Для обычного пользователя на Windows предусмотрен установщик
`installer/output/Jarvis_Setup.exe`. По умолчанию он ставит Jarvis Lite со
своим изолированным Python, Whisper, Silero и собственной NLU. Python и
PowerShell-команды вручную не нужны. Ollama и языковая модель не ставятся без
согласия пользователя: Jarvis Full можно выбрать отдельной галочкой или
включить позднее через ярлык в меню «Пуск». Инструкция по сборке установщика:
[`installer/README_RU.md`](installer/README_RU.md).

Ниже остаётся способ запуска из исходников для разработки.

Рекомендуется Python 3.10+ на Windows:

```powershell
cd "C:\Users\Hp Victus\Desktop\Jarvis"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py               # постоянный режим, Ctrl+Alt+Space для записи
python main.py --demo        # один цикл без микрофона и глобальной клавиши
```

Перед первым запуском и после изменения окружения выполните безопасную
runtime-диагностику:

```powershell
python main.py --doctor
python main.py --doctor --json   # структурированный отчёт для установщика/CI
```

Doctor ничего не скачивает, не открывает аудиопоток, не включает микрофон и не
воспроизводит звук. Он проверяет Python и Windows, RAM и диск, доступность
runtime-путей, собственный NLU checkpoint, PyTorch/CUDA/GPU, Whisper и его
локальный checkpoint, push-to-talk/VAD, устройства ввода и вывода, Silero TTS,
сервер Ollama и выбранную модель.

- `OK` — компонент готов;
- `WARN` — Jarvis запустится с ограничением или fallback;
- `FAIL` — настроенная основная возможность не заработает;
- `SKIP` — проверка неприменима или зависит от отсутствующего компонента.

Код возврата `0` означает отсутствие критических ошибок (предупреждения
допустимы), код `2` — наличие хотя бы одного `FAIL`. Каждая проблема содержит
конкретное действие. `--doctor` загружается отдельно от runtime-движков, поэтому
может диагностировать даже сломанный импорт PyTorch или аудиобиблиотеки.
Проверка аудиоустройств подтверждает наличие и выбор устройства по умолчанию,
но намеренно не записывает и не воспроизводит звук; качество реального сигнала
остаётся отдельной пользовательской проверкой.

Чтобы Whisper использовал NVIDIA GPU, после основных зависимостей установите
CUDA-вариант PyTorch (эта команда проверена с CUDA 12.8):

```powershell
python -m pip install --upgrade --force-reinstall torch `
  --index-url https://download.pytorch.org/whl/cu128
```

Проверка должна вывести `True` и название видеокарты:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Без CUDA-варианта PyTorch настройка `device: auto` безопасно выберет CPU.
Checkpoint Whisper `small` (около 461 МБ) скачивается только при первом запуске
в `models/openai-whisper/` и не добавляется в Git.

Команду также можно передать текстом, полностью минуя микрофон и STT:

```powershell
python main.py --text "открой калькулятор"
python main.py --text "какие приложения ты можешь открыть"
python main.py --text "сколько сейчас времени"
python main.py --text "через 10 минут напомни проверить духовку"
python main.py --text "напомни завтра в 18:30 позвонить маме"
python main.py --text "покажи мои напоминания"
python main.py --text "отмени напоминание номер 1"
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
Оно обучает с нуля собственную символьную CharCNN без Hugging Face. Модель
учится как менеджер: сначала различать маршруты `tool / control / dialogue /
reject`, затем выбирать конкретный intent и извлекать слоты. Runner сравнивает
`standard`, `augmented` и `curriculum`, контролирует старые команды, worst-class
recall и CPU latency, а затем проверяет победителя на двух holdout-наборах.
Экспорт разрешается только вместе с `approved.json` и контрольной SHA-256.

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
modules/    wake word, STT, собственный NLU, LLM, TTS, планировщик
ml/nlu/     датасет, токенизатор, модели, обучение, inference
models/     checkpoints и метрики экспериментов
tools/      автоматически обнаруживаемые инструменты
memory/     краткосрочная память и SQLite-хранилище
tests/      unit, regression и end-to-end тесты
```

Конфигурация модулей находится в `config.yaml`. Whisper `small` автоматически
выбирает CUDA/FP16, когда установлен CUDA-вариант PyTorch и доступна NVIDIA GPU;
с CPU-вариантом PyTorch он запускается на CPU/FP32. Собственная NLU работает на
CPU, а GPU-доступ моделей сериализуется через `GPULock`.

## Пока не реализовано

- постоянное прослушивание и настоящее распознавание фразы активации;
- подтверждённая на реальном микрофоне/колонках работа всей голосовой цепочки;
- закрытие приложений и управление их интерфейсом после запуска;
- использование долгосрочной памяти в диалоге;
- полностью neural exact slot extraction без constrained decoder;
- проверка всей цепочки barge-in на реальном микрофоне и аудиоустройстве.
