# Jarvis — checkpoint и handoff от 13 августа 2026

Этот документ — точка продолжения для нового чата Codex. Он описывает состояние
ветки `codex/parakeet-shadow-nlu-test` относительно предыдущего checkpoint
`607e2cb` от 9 августа 2026.

## Короткий статус

- Production runtime по-прежнему использует `openai-whisper small`.
- Parakeet TDT 0.6B v3 установлен и работает локально только как no-action
  shadow-кандидат; в `main.py` он не подключён.
- Production NLU усилена semantic commit gate, каноническим application
  resolver и атомарной обработкой составных команд.
- Локальный Jester Tiny3D подключён к Gesture Core в ограниченном режиме.
- Формальный Phase 2/голосовой benchmark не начинался.
- Fixture recording не начинался. В ожидаемом локальном каталоге
  `.local/parakeet/fixtures/phase_1_5` на момент checkpoint нет fixture root,
  manifest или WAV-файлов.
- Ни одна команда Jarvis во время Parakeet live-тестов не выполнялась.

## Что изменено после предыдущего checkpoint

### 1. Parakeet STT shadow

- Модель: `nvidia/parakeet-tdt-0.6b-v3`.
- Revision: `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`.
- Лицензия: CC-BY-4.0, принятие владельцем хранится как отдельное immutable
  local evidence и не связано с fixture consent.
- Локальная модель установлена в `.local/parakeet/models/`; этот путь исключён
  из Git.
- Windows runtime использует Transformers, CUDA и FP16 в отдельном persistent
  worker. Worker прогревается один раз, не получает доступ к runtime Jarvis и
  перезапускается после decode timeout.
- Live test принимает микрофон, WAV или текст и печатает JSON. Все candidate
  actions имеют `execution: blocked`.

Безопасная ручная проверка:

```bat
SETUP_PARAKEET.cmd --status
TEST_PARAKEET_NLU.cmd --mic
```

Не нужно запускать `main.py`: Parakeet в production STT не интегрирован.

### 2. Production NLU safety

Добавлен общий semantic commit gate перед production NLU и в shadow pipeline:

- `wait` — незаконченная команда, например `открой` или `запусти телегу и`;
- `rejected` — отрицание, цитата/обсуждение команды или самоотмена;
- `analyze` — завершённая команда;
- в shadow JSON итоговое безопасное состояние отображается как `ready`,
  `wait`, `clarify` или `rejected`.

Самокоррекция оставляет только последнюю мысль:

```text
Открой калькулятор. Нет, лучше блокнот. -> open_application(notepad)
Запусти телеграмм. Ой, закрой телеграмм. -> window_control(close, telegram)
```

Составная команда коммитится только целиком. Если хотя бы одна часть unknown
или ниже threshold, actions очищаются. Neural `open_application` не может
передать произвольный slot: значение обязано разрешиться через allow-list.

Подтверждённые aliases:

- VS Code: `Visual Studio Code`, `VS Code`, `вс код`, `вижуал студио код`,
  `визуал студио код`, `висуал студио код`, `вижу студио код`, `в скот`,
  `ваэс код`, `вэс код`, `код`;
- Telegram: `Telegram`, `телеграм`, `телеграмм`, `телега`;
- Discord: `Discord`, `дискорд`, `дискорт`, `дискот` и ранее разрешённые
  варианты.

### 3. Fixture schema/recorder

Код содержит canonical schema `jarvis.semantic_fixture.v1`. Каждая manifest row
должна включать ID, relative WAV path, SHA-256, speech flag, human reference,
semantic scoring flag, ordered actions, risk class, acoustic condition,
recording instructions и tags. Action shape совпадает с production:

```json
{
  "intent": "open_application",
  "slots": {
    "application": "calculator"
  }
}
```

Validator отклоняет старые intents (`open_app`, `search` и подобные), slot
`app`, некорректный WAV, небезопасный путь, неверный checksum и plan drift.
Recorder имеет `devices`, `test-device`, `record-next`, `record-all`, `replay`,
`re-record`, `delete`, `progress` и `validate`; перед записью крупно показывает
акустическое условие.

Это только готовый код. План не был сгенерирован заново, consent не запрашивался,
микрофон recorder не включался и benchmark не запускался.

### 4. Production Whisper

Whisper остаётся активным production STT. Его decoding настроен на короткие
команды: Russian language, deterministic temperature, beam search, отключённый
carry-over предыдущего текста и application prompt. Ошибки и malformed audio
теперь дают пустой transcript/confidence `0.0`, чтобы невозможна была
синтетическая команда из fallback.

### 5. Gesture Core

Локальный selected checkpoint:

```text
training_workspace/jester/runs/full/tiny_3d_cnn/best.pt
SHA-256: 0c14820042c652c8b51d19c82af57fd0f0706edcf317f1859cf2e461b80f7e07
```

Официальный test split содержит 14 743 клипа. Результат: accuracy 0.7922,
macro-F1 0.7871, negative recall 0.8962. Runtime принимает только ожидаемый
Tiny3D contract и отображает шесть выбранных Jester gestures в безопасные
`G01`–`G06`; все остальные сворачиваются в `D0X`.

Checkpoint, датасет и generated report локальны и исключены из Git. После
обычного clone этот runtime-артефакт нужно будет восстановить отдельно.

## Наблюдавшееся качество Parakeet

В корректирующем live-прогоне один владелец произнёс девять целевых реплик.
Все девять дали ожидаемый semantic outcome, timeout не возник. Это полезная
регрессия, но не общий STT benchmark.

По 23 успешным captures из предоставленных логов Parakeet имел median decode
1242.9 ms и p95 1740.2 ms. Исторический, но не парный прогон Whisper-small на
41 другой реплике имел median 718 ms и p95 3008 ms. Поэтому нельзя честно
утверждать, что Parakeet быстрее или точнее: аудио, фразы и выборки различались,
а human reference для вычисления WER/CER не подготовлен.

## Проверки checkpoint

Перед публикацией выполнены:

```powershell
.\venv\Scripts\python.exe -m pytest -q `
  --ignore=tests/test_ipn_tsn_data.py --ignore=tests/test_jester_pipeline.py
.\.venv-training\Scripts\python.exe -m pytest `
  tests/test_ipn_tsn_data.py tests/test_jester_pipeline.py -q
git diff --check
```

Результаты:

- runtime suite без двух training-only модулей: `449 passed, 2 skipped`;
- training-only IPN/Jester suite: `27 passed`;
- отдельная целевая STT/NLU/Parakeet/Gesture регрессия: `139 passed, 1 skipped`;
- `git diff --check`: ошибок whitespace нет, только ожидаемые Windows CRLF
  warnings при чтении рабочего дерева.

Первый полный сбор тестов через runtime `venv` остановился на collection двух
training-модулей из-за отсутствующего `tensorboard`. Это не скрывалось и не
исправлялось установкой лишней зависимости в production environment: эти тесты
успешно запущены через предназначенный для них `.venv-training`.

Model smoke, микрофон, `main.py` и реальные действия не входят в автоматическую
проверку checkpoint.

## Следующий разумный этап

1. Не менять production STT до парного сравнения на одном аудио.
2. Решить, проверять ли нативный Parakeet runtime (`parakeet.cpp` или
   sherpa-onnx Parakeet TDT) против текущего Transformers worker.
3. Отдельно утвердить fixture plan, consent и retention; только после этого
   записывать или запускать benchmark.
4. Для NLU собирать пары `искажённый STT-текст -> правильные ordered actions`
   вместе с negative/incomplete examples. Незаконченную мысль нельзя размечать
   как обычную ошибку STT.
5. После выбора корпуса обучать новый NLU candidate, не меняя frozen holdouts и
   не подключая его к production до semantic/false-positive gates.

## Границы приватности и Git

В Git не входят `.local/`, модели Parakeet, license acceptance evidence,
fixture consent, WAV, пользовательские логи, Jester checkpoints/datasets и
generated reports. Публикуются только код, schemas, documentation и tests.
