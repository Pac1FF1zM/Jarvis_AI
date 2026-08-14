# Jarvis — модульный локальный голосовой ассистент

Jarvis — событийный Python-ассистент для Windows. Модули не вызывают друг
друга напрямую: данные проходят через `EventBus`, а `Orchestrator` контролирует
состояния `IDLE → LISTENING → TRANSCRIBING → THINKING → TOOL_CALL → SPEAKING`.
Ключевые события имеют frozen dataclass-контракты из `core/event_payloads.py`:
EventBus проверяет обязательные поля и диапазоны, отклоняет лишние/ошибочные
поля известных событий и кладёт в очередь рекурсивно неизменяемый снимок.
Неизвестные события расширений остаются допустимыми, но их payload также
отделяется от словаря производителя перед публикацией.

## Статус проекта — 15 августа 2026

Текущий runtime использует единый production STT:

| Область | Текущее состояние |
|---|---|
| Основной STT | `nvidia/parakeet-tdt-0.6b-v3`, pinned local production worker |
| Предыдущий STT | Whisper удалён из runtime и зависимостей; сохранён только исторический benchmark |
| Semantic routing | `agreement_canary`: NLU — активный control/fallback, единый coordinator исключает двойное выполнение |
| Structured JSC | v8 seed29; независимый JAL candidate и agreement telemetry без side effects |
| Gesture Core | Jester Tiny3D release export включён в `models/`; restricted G01–G06 |
| JSC/JAL promotion | restricted/JSC-primary/NLU-removed пути реализованы, но fail-closed до human voice gates и 1/2 стабильных циклов |
| Голосовой benchmark | FLEURS `ru_ru`, 20 парных human-speech записей ≤12 с: Parakeet WER 4,40%, Whisper 5,35% |
| Полный pytest | 570 passed, 3 skipped в `.venv-training` |

Полный актуальный handoff: [статус проекта 2026-08-15](docs/PROJECT_STATUS_2026-08-14_RU.md).
Текущая матрица admission:
[JSC migration readiness](docs/evidence/JSC_MIGRATION_READINESS_20260814.json).
Предыдущий [checkpoint 2026-08-13](docs/CHECKPOINT_2026-08-13_RU.md) сохранён
как исторический снимок.
Отчёт о выборе STT: [production Parakeet](docs/PARAKEET_PRODUCTION_EXPERIMENT_RU.md).

## Что работает сейчас

`python main.py` запускает постоянный голосовой режим. Локальный openWakeWord
слушает фразу «Hey Jarvis»/«Jarvis»; глобальная комбинация `Ctrl+Alt+Space`
остаётся надёжным push-to-talk fallback. После акустического пробуждения Jarvis
случайно отвечает «К вашим услугам, сэр» или «Что прикажете делать?», а затем
Silero VAD записывает команду до окончания речи и запускается полный цикл:

```text
wake_word_detected
  → audio_captured
  → transcription_ready
  → nlu_result + jsc_candidate_ready
  → semantic_result (agreement canary coordinator)
  → tool_call_requested / general chat
  → response_ready
  → speech_started
  → speech_finished
  → IDLE
```

- Активация фразой выполняется локальной ONNX-моделью openWakeWord; её файлы
  загружаются один раз при первом голосовом запуске. Два последовательных
  положительных аудиокадра снижают случайные срабатывания. Захват микрофона
  настоящий: `sounddevice`, mono PCM16, 16 кГц, с Silero VAD и ограничением
  максимальной длительности. Без wake-word модели остаётся горячая клавиша,
  без аудиобиблиотек — безопасный `python main.py --demo`.
- Калибровка микрофона привязана к профилю пользователя и конкретному
  аудиоустройству. Она настраивает пороги начала/окончания речи и безопасное
  усиление PCM; после смены микрофона Jarvis использует стандартные значения.
- Production STT — многоязычный `nvidia/parakeet-tdt-0.6b-v3` с закреплённой
  revision. При доступной NVIDIA CUDA worker использует FP16, иначе доступен
  явный CPU-режим. Ошибка worker, timeout или некорректное аудио дают пустой
  transcript: production-путь не подставляет фиктивную команду.
- NLU — собственная нейросеть проекта, обученная с нуля, вместе с узким
  детерминированным router. Перед ними работает semantic commit gate:
  незаконченные, отрицающие и процитированные команды блокируются,
  самокоррекция оставляет только последнюю мысль, а составной план принимается
  атомарно — одна неизвестная часть запрещает все действия.
- Structured JSC v8 получает тот же transcript параллельно с NLU, строит
  типизированный JAL, хранит history/pending state и публикует только
  неисполняемый candidate. Единый migration coordinator в активном
  `agreement_canary` передаёт дальше NLU и пишет каноническое сравнение в
  `logs/jsc_agreement.jsonl`; поэтому два semantic-пути не могут одновременно
  выполнить команду. Migration development Exact JAL — 87,75%, а свежий
  офлайн runtime gate — 96,75%; оба результата не заменяют human voice gate.
- Restricted reversible и JSC-primary execution уже реализованы отдельным JAL
  executor: schema validation, completeness/calibrated abstention, correction
  compensation, stop-on-failure и compound rollback. Runtime не допускает эти
  стадии по одному config-флагу: нужны versioned human metrics; NLU удаляется
  только после двух стабильных JSC-primary release-циклов.
- Ollama используется только для свободного диалога. Решения о запуске
  инструментов принимает выбранный semantic owner; в активном canary это NLU,
  а JSC не имеет side effects. Готовый результат возвращается напрямую;
  при недоступности Ollama работает текстовая заглушка. Доступность локальной
  модели проверяется во время запуска, поэтому первый вопрос не ждёт сетевого
  тайм-аута уже выключенного сервера.
- TTS настроен на русский Silero `v4_ru` (`xenia`, 48 кГц) и `sounddevice`;
  без них работает заглушка. Голосовые движки загружаются параллельно, ленивые
  CPU-ядра Silero прогреваются в отслеживаемой фоновой задаче, а короткие
  повторяющиеся ответы берутся из ограниченного аудиокэша.
- Долговременная память автоматически выделяет только важные профильные факты:
  имя, возраст, учёбу, работу, цель и город. Явные команды «запомни…», «что ты
  обо мне знаешь?» и «забудь…» также работают без Ollama. Для каждого профиля
  хранится не более пяти популярных приложений. Обычный контекст очищается при
  перезапуске; читаемая история чатов хранится отдельно и не подмешивается в
  новую сессию. Полная очистка памяти доступна голосом и из Control Center.
- NLU понимает разные формулировки запуска, выключения, паузы, продолжения и
  проверки жестового режима. После голосового запуска Jarvis произносит
  «Жестовый режим активирован», затем Control Center автоматически открывает
  встроенную вкладку камеры и телеметрии; голосовое управление продолжает
  работать. `Ctrl+Alt+/` включает и выключает режим, пока запущен Jarvis.
  При developer-запуске без Control Center отдельное окно остаётся резервным
  вариантом; после перезапуска режим всегда выключен.
- Локально обученный с нуля Jester Tiny3D получил accuracy 0.7922 и macro-F1
  0.7871 на официальном test split из 14 743 клипов. Runtime сворачивает 27
  классов в шесть обратимых media-действий (`G01`–`G06`) и `D0X`; остальные
  классы не могут стать действием. Модель остаётся в режиме ограниченной
  real-camera проверки, а не финально одобренным production-классификатором.
- Телеметрия жестов пишется в локальные JSONL-файлы `logs/gestures` без кадров
  и видео. Хранение ограничено 30 днями и суммарно 100 МБ.
- `python main.py --gesture_mode` запускает Gesture Core отдельно: только
  webcam, нейросеть, шесть тестовых media-действий и live-preview, без STT,
  NLU, LLM, TTS, wake word и оркестратора. На Windows камера открывается через
  DirectShow; закрытие preview не прекращает распознавание, `Ctrl+C` завершает
  процесс и освобождает устройство.
  Для окна следует запускать runtime-окружение `venv`: обучающее
  `.venv-training` использует headless OpenCV и предназначено только для обучения.
- Необработанное исключение handler превращается в `interaction_failed`;
  watchdog возвращает зависший trace в `IDLE`, а запоздавшие события закрытого
  trace игнорируются. Recovery также останавливает захват и TTS.
- Повторная активация во время записи, STT, размышления, инструмента или речи
  атомарно отменяет старый trace и передаёт владение новому. Команда «стоп»
  завершает текущий trace без дополнительной озвучки. Асинхронный инструмент
  отменяется, а неотменяемый worker STT/LLM безопасно дренируется: его поздний
  результат уже не может открыть приложение, заговорить или изменить state.
- Доступны время, запуск приложений, окна, пользовательские файлы, настройки
  Windows, браузерный поиск/сайты/вкладки, громкость и управление музыкой.
  Составная команда, включая строку STT без запятых, превращается в
  последовательный план под одним trace; числа «четырнадцать», «двадцать» и
  составные числительные нормализуются в параметры времени. Поддерживаются
  местоимения «закрой его», отрицания, исправления и безопасная команда
  «отмени последнее действие». Если параметра напоминания не хватает, Jarvis
  задаёт уточняющий вопрос и продолжает план после ответа;
  если одна часть не разобрана, side effects не начинаются. Удаление отправляет
  объект в корзину, а опасные системные операции требуют отдельного голосового
  подтверждения. Напоминания сохраняются в SQLite, переживают перезапуск Jarvis,
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

В собственной NLU/JSC-части Hugging Face, готовые веса, готовые embeddings,
внешние токенизаторы и скачанные датасеты не используются. PyTorch служит
только вычислительным фреймворком. Parakeet, Silero и openWakeWord являются
отдельными локальными аудиодвижками, а не «мозгом» Jarvis. Whisper в
production runtime больше не используется.

Выбранный runtime checkpoint — `models/nlu_manager_finetuned.pt`. Это
`CharCNN + augmented` со 169 199 параметрами:

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

### Gesture ML: IPN Hand и Jester

В `src/` находится отдельный воспроизводимый pipeline распознавания жестов:
аудит IPN Hand, subject-disjoint split, TSN/3D-модели, обязательные gates,
обучение, оценка и inference по изолированному видео. Обучающие зависимости
закреплены в `training_workspace/requirements-training.txt`; скачанные видео,
checkpoints, кэш и generated-отчёты в Git не добавляются.

Для отдельного обучения динамической модели с нуля подготовлен Jester pipeline:
три случайно инициализированных кандидата, потоковая распаковка официального
multipart-TGZ, равнобюджетный benchmark и закрытый до финала test split.
Инструкции находятся в `training_workspace/jester/README.md`, зависимости — в
`training_workspace/requirements-jester.txt`, окружение создаёт
`SETUP_JESTER_TRAINING.cmd`.

В текущем runtime выбран `tiny_3d_cnn`; загрузчик проверяет тип checkpoint,
полный порядок 27 меток, preprocessing-параметры и SHA-256. Выбранный компактный
release-checkpoint и финальный quality report находятся в
`models/gesture/20260812_jester_tiny3d/` и входят в Git/Setup. Тяжёлые датасеты,
полные training runs и промежуточные checkpoints остаются локальными.

Проверка отдельного видео с выбранным TSN checkpoint:

```powershell
.\.venv-training\Scripts\python.exe -m src.infer video.mp4 --checkpoint checkpoints/tsn_resnet18_seed42/best.pt
```

Для адаптации к собственной камере автоматический recorder показывает класс,
делает обратный отсчёт, записывает трёхсекундный клип и сам создаёт разметку:

```powershell
.\.venv-training\Scripts\python.exe -m src.record_custom_dataset --split train
.\.venv-training\Scripts\python.exe -m src.record_custom_dataset --split val
.\.venv-training\Scripts\python.exe -m src.record_custom_dataset --split test
```

Train, validation и test сохраняются раздельно в `data/custom_capture/`.
Управление и короткий режим проверки описаны в
[`docs/custom_dataset_recording.md`](docs/custom_dataset_recording.md).

Запуск приложений требует явной командной конструкции. Системные цели имеют
фиксированные безопасные обработчики, а остальные приложения обнаруживаются
только в меню «Пуск» и Windows `App Paths`. Jarvis открывает зарегистрированный
shortcut/исполняемый файл напрямую и никогда не превращает распознанную речь в
строку PowerShell, CMD или произвольную shell-команду.

## Parakeet production и безопасный shadow-тест

Parakeet TDT 0.6B v3 является единственным production STT. Его
закреплённый checkpoint работает в отдельном постоянном процессе, принимает тот
же PCM 16 кГц mono, что выдаёт microphone pipeline, и публикует стандартный
`transcription_ready`. Ошибка или timeout дают пустой transcript; скрытого
fallback и смешивания гипотез нет.

Отдельный no-action shadow-контур сохранён для безопасной диагностики:

```bat
SETUP_PARAKEET.cmd --status
TEST_PARAKEET_NLU.cmd --mic
```

В live-режиме Enter начинает и завершает одну реплику. Каждое распознанное
действие помечено `execution: blocked`. В отличие от него, обычный `main.py`
является реальным Jarvis runtime и может выполнять прошедшие safety-гейты
команды. Модель, immutable evidence принятия
лицензии и любые будущие private fixtures находятся в `.local/` и не попадают
в Git. Установка и диагностика подробно описаны в
[experiments/parakeet/README.md](experiments/parakeet/README.md).

## Установка и запуск

### Jarvis Control Center — основной способ управления

Jarvis теперь запускается и настраивается через локальное desktop-приложение.
В нём доступны:

- запуск и штатная остановка runtime с отображением состояния загрузки;
- основные настройки голоса, STT, LLM и Gesture Core без редактирования YAML;
- расширенный YAML-редактор с проверкой перед атомарным сохранением;
- полный `--doctor` в виде таблицы PASS/WARN/FAIL с рекомендациями;
- пошаговая калибровка голоса кнопками, без сохранения сырых записей;
- встроенный жестовый режим с камерой, TOP-3, уверенностью и производительностью;
- рабочие пространства с визуальной схемой окон, захватом текущего расположения,
  файлами, сайтами и временным виртуальным рабочим столом Windows;
- редактируемые стартовые режимы «Программирование», «Игры» и «Учёба»;
- вкладка диалогов с автоматически выбранными названиями, сортировкой по дате,
  важными фактами профиля и отдельными кнопками очистки памяти и истории;
- живой журнал запуска и работы Jarvis.

Режим «Программирование» открывает VS Code с последним проектом и браузер;
ChatGPT включается пользователем как необязательный ресурс. «Игры» запускает
Steam, Discord и браузер, а в редакторе предлагаются Spotify, Epic Games
Launcher и OBS Studio. «Учёба» оставляет браузер и Telegram, открывает ChatGPT,
Claude и DeepSeek; найденные игры закрываются только после подтверждения.
Лишние пользовательские окна Jarvis не закрывает. Команда выхода удаляет только
созданный Jarvis временный рабочий стол и оставляет приложения открытыми.

Для запуска из текущего рабочего проекта дважды нажмите
`START_JARVIS_UI.cmd` или выполните:

```powershell
.\venv\Scripts\pythonw.exe jarvis_control.py
```

Дальнейшие пользовательские возможности подключаются к Control Center.
`main.py` остаётся внутренней runtime-точкой и консольным fallback для тестов и
разработки. Привязка Control Center к устанавливаемому `.exe` будет выполнена
отдельным этапом после стабилизации интерфейса.

Для обычного пользователя на Windows предусмотрен установщик
`installer/output/Jarvis_Setup.exe`. По умолчанию он ставит Jarvis Lite со
своим изолированным Python, Parakeet runtime, Silero и собственной NLU. Python и
PowerShell-команды вручную не нужны. Ollama и языковая модель не ставятся без
согласия пользователя: Jarvis Full можно выбрать отдельной галочкой или
включить позднее через ярлык в меню «Пуск». Инструкция по сборке установщика:
[`installer/README_RU.md`](installer/README_RU.md).

Ниже остаётся консольный способ запуска только для разработки и диагностики.

Рекомендуется Python 3.10+ на Windows:

```powershell
cd "C:\Projects\Jarvis"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python jarvis_control.py     # основной desktop-интерфейс
python main.py               # внутренний постоянный runtime
python main.py --demo        # один цикл без микрофона и глобальной клавиши
```

### Профиль и калибровка голоса

Перед первым голосовым запуском откройте в Control Center раздел
«Калибровка», остановите runtime и пройдите пять записываемых этапов. Команды
ниже сохранены только как developer fallback:

```powershell
python main.py --profiles
python main.py --calibrate-voice
python main.py --calibrate-voice --profile mikhail --profile-name "Михаил"
```

Мастер запишет четыре секунды фоновой тишины, три короткие фразы обычным,
тихим и громким голосом, а затем проверочную фразу. Если сигнал слишком шумный
или перегруженный, ненадёжный результат не сохраняется. Повторный запуск с
другим микрофоном добавляет отдельную калибровку, не уничтожая предыдущую.

Профили находятся вне проекта и будущего `.exe`: по умолчанию в
`%APPDATA%\Jarvis\profiles`, а при заданном `JARVIS_DATA_DIR` — в
`%JARVIS_DATA_DIR%\profiles`. Сырые аудиозаписи не сохраняются: только уровни,
пороги VAD, усиление и отпечаток устройства. Файл `speech_aliases.json` внутри
профиля сохраняет личные варианты произношения иностранных названий для
совместимости; текущий Parakeet runtime не использует prompt-подсказки.
Калибровка улучшает условия записи, но не
идентифицирует человека по голосу.

Факты долговременной памяти сохраняются только после явной команды, например:

```text
Запомни, что меня зовут Алексей.
Что ты обо мне знаешь?
Как меня зовут?
Забудь Алексей.
Забудь всё.               # Jarvis запросит подтверждение
```

Память переживает перезапуск, но не является зашифрованным хранилищем секретов:
пароли, API-ключи и платёжные данные сохранять в ней не следует.

Перед первым запуском и после изменения окружения выполните безопасную
runtime-диагностику:

```powershell
python main.py --help
python main.py --doctor
python main.py --doctor --json   # структурированный отчёт для установщика/CI
```

Doctor ничего не скачивает, не открывает аудиопоток, не включает микрофон и не
воспроизводит звук. Он проверяет Python и Windows, RAM и диск, доступность
runtime-путей, собственный NLU checkpoint, PyTorch/CUDA/GPU, выбранный STT и
его локальный checkpoint, push-to-talk/VAD, устройства ввода и вывода, Silero TTS,
сервер Ollama, выбранную модель и совместимость активной калибровки с текущим
микрофоном.

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

Чтобы Parakeet использовал NVIDIA GPU, после основных зависимостей установите
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
Pinned checkpoint Parakeet скачивается только после просмотра и явного принятия
CC-BY-4.0 через `SETUP_PARAKEET.cmd`; веса в `.local/parakeet/models/` не
добавляются в Git.
Модели openWakeWord для фразы активации также загружаются только один раз; если
это не удалось, запуск не ломается и продолжает работать через `Ctrl+Alt+Space`.
В сонном режиме Jarvis реагирует на фразу `Hey Jarvis` или горячую клавишу.
После ответа он автоматически слушает следующую команду ещё 7 секунд. Каждая
успешная следующая реплика продлевает активную сессию, а тишина возвращает
ассистента в сонный режим после голосового сигнала «Отключаюсь». Обычные окна,
включая браузер,
Telegram и Discord, закрываются сразу; подтверждение остаётся для удаления
файлов, блокировки компьютера и других потенциально опасных действий.
Отсчёт активной сессии начинается только после завершения инструмента и речи
Jarvis. Как только VAD обнаружил начало ответа пользователя, отсчёт прекращается
и не может оборвать длинную голосовую команду до публикации записанного аудио.

Команду также можно передать текстом, полностью минуя микрофон и STT:

```powershell
python main.py --text "открой калькулятор"
python main.py --text "какие приложения ты можешь открыть"
python main.py --text "сколько сейчас времени"
python main.py --text "через 10 минут напомни проверить духовку"
python main.py --text "напомни завтра в 18:30 позвонить маме"
python main.py --text "покажи мои напоминания"
python main.py --text "отмени напоминание номер 1"
python main.py --text "найди в интернете погоду в Ташкенте"
python main.py --text "открой настройки звука"
python main.py --text "найди файл диплом"
python main.py --text "открой браузер и скажи время"
python main.py --text "нет, я имел в виду Discord"
```

Текст проходит через ту же собственную NLU-модель, оркестратор и инструменты;
пропускаются только симулированные захват аудио и STT.

Каждый запуск автоматически сохраняет отдельный UTF-8 лог в
`logs/sessions/jarvis_session_*.txt`. Путь печатается при старте как
`SESSION_LOG_READY`. Файл содержит распознанный STT-текст, confidence,
решение NLU, параметры инструмента, ответ и ошибки; его можно прикладывать как
feedback для подготовки следующего обучающего корпуса. `logs/jarvis.log`
по-прежнему содержит только последний запуск для анализа задержек.

### Active Learning из реальных ошибок

Помимо обычного лога Jarvis локально кладёт только сомнительные NLU-решения
(`unknown`, низкая уверенность) и неудачные выполнения в
`data/feedback/pending.jsonl`. Аудио, ключи, LLM-переписка и ответы инструментов
туда не попадают. Очередь ограничена 5 МБ, не добавляется в Git и **никогда не
используется для автоматического переобучения**.

Перед следующим обучением человек размечает кандидаты:

```powershell
python -m training_workspace.review_feedback --summary
python -m training_workspace.review_feedback
```

Только подтверждённые записи попадают в
`training_workspace/data/feedback_train.jsonl`; runner подключит их к train,
сохраняя validation и оба holdout независимыми.

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
reject`, затем выбирать конкретный intent и извлекать слоты. Runner выполняет
двухэтапный поиск: 24 конфигурации сравниваются с одинаковым seed, а три
финалиста повторяются на пяти seed. Отбор учитывает intent, slots, точную
semantic frame, ложные slots, калибровку и CPU latency. Два holdout открываются
только после выбора одного репрезентативного checkpoint.
Экспорт разрешается только вместе с `approved.json` и контрольной SHA-256.

На компьютере с RTX 3090:

```powershell
.\training_workspace\SETUP_RTX3090.ps1
.\training_workspace\START_TRAINING.ps1 -CheckOnly
.\training_workspace\START_TRAINING.ps1
```

Сохранённые checkpoints можно повторно оценить после исправления decoder без
нового обучения:

```powershell
.\training_workspace\START_TRAINING.ps1 -ReevaluateRun "C:\path\nlu_report.json"
```

Формат данных, правила независимого holdout, интерпретация отчёта и безопасное
подключение победителя описаны в
[`training_workspace/GUIDE_RU.md`](training_workspace/GUIDE_RU.md).

## Проверка

Runtime и training pipeline используют разные закреплённые окружения:

```powershell
.\venv\Scripts\python.exe -m pytest -q `
  --ignore=tests/test_ipn_tsn_data.py --ignore=tests/test_jester_pipeline.py
.\.venv-training\Scripts\python.exe -m pytest `
  tests/test_ipn_tsn_data.py tests/test_jester_pipeline.py -q
```

Тесты включают полный событийный цикл, несколько последовательных голосовых
циклов, мокированный микрофон и VAD, настоящий NLU checkpoint, сохранение
`trace_id`, выполнение NLU/STT/TTS вне event-loop потока и конкурентную отмену
TTS. Отдельные adversarial-тесты проверяют двойную активацию, отмену долгого
инструмента и завершение блокирующего LLM-worker без stale-ответа.

Исторические реальные голосовые сессии можно обезличенно проанализировать и
сравнить с кандидатами pipeline так:

```powershell
.\.venv-training\Scripts\python.exe -m training_workspace.run_voice_acceptance `
  --logs logs/sessions --since 20260807
```

Методика и сценарии нового живого прогона описаны в
[`docs/VOICE_ACCEPTANCE_RU.md`](docs/VOICE_ACCEPTANCE_RU.md).

## Структура

```text
core/       EventBus, Orchestrator, GPULock, конфигурация
modules/    wake word, STT, NLU/JSC coordinator, JAL executor, LLM, TTS
ml/nlu/     legacy control/fallback, датасет, обучение, inference
ml/jsc/     JAL schema, Structured JSC, migration gates и transactions
experiments/ изолированные no-action STT/semantic-кандидаты
models/     checkpoints и метрики экспериментов
tools/      автоматически обнаруживаемые инструменты
memory/     краткосрочная память и SQLite-хранилище
tests/      unit, regression и end-to-end тесты
```

Конфигурация модулей находится в `config.yaml`. STT использует только pinned
Parakeet worker; `model_dir`, отдельный Python и timeout задаются в
`modules.stt.params`. Legacy NLU-control и Structured JSC работают на CPU, а
GPU-доступ моделей сериализуется через `GPULock`.

## Пока не реализовано

- парный benchmark на приватном корпусе именно боевых команд владельца (публичный
  FLEURS smoke уже выполнен, но не заменяет этот acceptance gate);
- фактический promotion выше agreement canary: runtime готов, но ещё нужны
  1 000 reviewed human voice turns, frozen holdout и correction/OOD gates;
- накопление свежей live seed29 telemetry; офлайн evidence уже зафиксирован,
  но намеренно не выдаётся за production-наблюдение;
- произвольная автоматизация интерфейса внутри любого приложения (клики по
  неизвестным кнопкам, заполнение форм и чтение содержимого экрана);
- полностью neural exact slot extraction без constrained decoder;
- проверка всей цепочки barge-in на реальном микрофоне и аудиоустройстве.
