# Обучение собственной NLU-модели-менеджера Jarvis на RTX 3090

Этот workspace обучает с нуля компактный символьный `CharCNN`-менеджер Jarvis.
Он не загружает модели, embeddings, токенизаторы или датасеты Hugging Face.
PyTorch используется только как вычислительный фреймворк.

Модель является мозгом маршрутизации, а не генератором текста: она выбирает
маршрут `tool / control / dialogue / reject`, затем конкретный intent и слоты.
Точные ответы времени и приложений формируют детерминированные инструменты;
свободный разговор после маршрутизации выполняет Ollama.

## Что делает один запуск

`START_TRAINING.ps1` последовательно:

1. проверяет JSONL, отсутствие пересечения train/validation, два holdout и CUDA;
2. измеряет baseline только на development-наборах, не раскрывая holdout;
3. запускает 24 целевых `augmented`-конфигурации вокруг лучшего семейства
   прошлого поиска с одним seed, чтобы сравнение параметров было честным;
4. оставляет четыре лучших конфигурации и повторяет каждую на пяти seed с полным
   бюджетом эпох; неустойчивый «везучий» запуск не становится победителем;
5. использует символьный токенизатор, поэтому новые слова не превращаются в
   один бесполезный `<unk>`;
6. балансирует не только intents, но и долю старого/нового корпуса, не позволяя
   новым шаблонам вытеснить старые навыки;
7. добавляет иерархический route-loss, штраф за слоты, несовместимые с intent,
   и frame-level no-slot loss против хотя бы одного ложного аргумента;
8. применяет AMP, TF32, gradient clipping, label smoothing, warmup + cosine
   decay, EMA весов, калибровку и stability-aware early stopping по macro-F1,
   worst-class recall, slots и полной semantic frame;
9. проверяет intent macro-F1, worst recall, F1 каждого slot, hallucination rate,
   semantic-frame exact match, end-to-end accuracy, ECE и CPU latency;
   neural BIO-метрики считаются отдельно до regex fallback, поэтому guardrails
   не могут скрыть слабую обученную модель;
10. только после выбора одного репрезентативного checkpoint один раз открывает
    два holdout и создаёт `approved.json` лишь после прохождения всех гейтов.

RTX 3090 ускорит серию экспериментов, но manager всё равно содержит лишь
десятки или сотни тысяч параметров. Главный источник роста точности — разнообразные и правильно
размеченные данные, а не загрузка 24 ГБ VRAM.

## 1. Подготовка компьютера друга

Нужны Windows 10/11, свежий драйвер NVIDIA, Python 3.10–3.12 и весь каталог
проекта Jarvis. Откройте PowerShell в корне проекта.

Создайте отдельное окружение и установите CUDA-сборку PyTorch:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\training_workspace\SETUP_RTX3090.ps1
```

По умолчанию используется официальный wheel-канал `cu128`. Если актуальный
селектор PyTorch рекомендует другой канал, передайте его явно, например:

```powershell
.\training_workspace\SETUP_RTX3090.ps1 -CudaWheel cu126
```

Актуальный канал всегда сверяйте с официальным селектором:
https://docs.pytorch.org/get-started/locally/

В конце должно быть:

```text
available= True
gpu= NVIDIA GeForce RTX 3090
```

## 2. Подготовка данных

В репозитории уже подготовлен сбалансированный датасет v3:

- `train.jsonl`: 1120 примеров, по 160 на каждый intent;
- `validation.jsonl`: 210 примеров, по 30 на каждый intent;
- `evaluation_holdout.jsonl`: 105 примеров, по 15 на каждый intent;
- `ml/nlu/holdout_v2.jsonl`: второй замороженный holdout из 49 фраз.

Он включает все приложения allow-list, русские/английские варианты названий,
реальные и безопасные ошибки Whisper, вежливые конструкции, отрицательные
контрасты и размеченные слоты. Подробности находятся в `data/README_RU.md`.
Пересобирать датасет перед обучением не требуется. Для проверки
воспроизводимости можно выполнить:

```powershell
python -m training_workspace.build_dataset
```

Редактируйте файлы:

- `data/train.jsonl` — примеры, на которых обновляются веса;
- `data/validation.jsonl` — отдельные фразы для выбора эпохи и эксперимента;
- `data/evaluation_holdout.jsonl` — финальная проверка, которую нельзя
  использовать при подборе настроек.
- `ml/nlu/holdout_v2.jsonl` — независимая последняя проверка уже выбранного
  кандидата; не редактируйте её после запуска экспериментов.

Одна строка — один JSON-объект:

```json
{"text":"открой калькулятор","intent":"open_application","slots":{"application":"калькулятор"}}
{"text":"через 12 минут напомни проверить духовку","intent":"set_reminder","slots":{"duration":"12","reminder_text":"проверить духовку"}}
{"text":"расскажи про космос","intent":"general_chat","slots":{}}
```

Допустимые intents:

- `get_current_time`
- `set_reminder`
- `open_application`
- `list_applications`
- `cancel`
- `general_chat`
- `unknown`

Допустимые slots: `duration`, `reminder_text`, `application`. Значение slot
должно буквально встречаться в `text`; loader сам вычислит символьные границы.

Правила дальнейшего расширения корпуса:

- сохраняйте минимум 100–200 уникальных train-фраз на intent;
- держите классы примерно сбалансированными;
- validation — минимум 30–50 новых фраз на intent, не перефразированные копии;
- добавляйте реальные ошибки Jarvis, опечатки и типичные результаты STT;
- для `unknown` используйте неполные/неразборчивые команды, а не нормальные
  вопросы, которые должны идти в `general_chat`;
- не копируйте validation или evaluation holdout в train после просмотра результата;
  для следующего цикла создавайте новый holdout;
- application должен соответствовать безопасному allow-list Jarvis. Новое имя
  в NLU не добавляет программу в allow-list автоматически.

Напоминания сейчас честно отклоняются из-за отсутствия постоянного scheduler.
Fine-tuning улучшит распознавание intent/slots, но не создаст scheduler.

## 3. Добавление проверенного feedback из реальных сессий

Jarvis сохраняет только кандидаты с низкой уверенностью, `unknown` и неудачные
выполнения в локальную очередь `data/feedback/pending.jsonl`. Нельзя обучаться
на ней напрямую: сначала проверьте статистику и разметьте записи вручную:

```powershell
python -m training_workspace.review_feedback --summary
python -m training_workspace.review_feedback
```

Подтверждённые примеры будут записаны в
`training_workspace/data/feedback_train.jsonl`; runner автоматически добавит
их только к train. Validation и holdout остаются нетронутыми. Если в ответе
Jarvis была личная информация, запись лучше пометить `d` (discard).

## 4. Проверка до долгого запуска

```powershell
.\training_workspace\START_TRAINING.ps1 -CheckOnly
```

Команда проверит пути, JSON, intents, slots и статистику классов без обучения.

## 5. Запуск

Перенесите на компьютер друга весь актуальный проект. Для обучения обязательно
нужны как минимум:

- `models/nlu_word_bigru_curriculum.pt` — проверенный базовый checkpoint;
- весь каталог `training_workspace`, включая новый `data/`;
- каталоги `ml/nlu` и корневые Python-зависимости проекта.

Не используйте старый `training_workspace/export/jarvis_nlu_best.pt`, который
был получен на демонстрационном наборе из 14 train-примеров. Базовый checkpoint
нужен для контрольных метрик; новые manager-кандидаты обучаются с нуля. Старый
файл в `export/` невозможно скопировать без свежего `approved.json` и совпадения
SHA-256.

```powershell
.\training_workspace\START_TRAINING.ps1
```

Настройки экспериментов находятся в `config.yaml`. На RTX 3090 можно сначала
оставить `batch_size: 256`. Если возникает CUDA OOM, уменьшайте до 128/64; если
GPU почти пуст и данных стало много — увеличивайте до 512.

Не закрывайте PowerShell. Первый этап обучает 24 коротких trial; второй — четыре
финалиста на seed `17, 43, 101, 211, 307`. Каждая эпоха печатает loss, общий
manager score, intent F1, worst recall, slot F1, semantic-frame exact match и
hallucination rate. Early stopping завершает бесполезные эпохи автоматически.

## 6. Результаты

Каждый запуск создаёт:

```text
training_workspace/runs/YYYYMMDD_HHMMSS/
  search_01.pt ... search_24.pt
  search_XX_seed_17.pt ... search_XX_seed_307.pt
  *.metrics.json
  report.json
```

Если конфигурация устойчиво прошла минимум четыре из пяти seed и финальный
checkpoint прошёл joint-quality, regression, calibration и latency-гейты,
победитель появится здесь:

```text
training_workspace/export/jarvis_nlu_best.pt
training_workspace/export/approved.json
```

`COPY_BEST_TO_MODELS.ps1` принимает checkpoint только при наличии свежего
`approved.json` с совпадающим SHA-256. Если approval не появился, это нормальная
защита: модель не прошла regression/holdout. Изучите `report.json`; не понижайте
пороги только ради появления файла.

Если обучение уже завершилось, а изменился только evaluator/decoder, веса не
нужно обучать повторно. Пока исходный отчёт не содержит раздел `holdouts` и все
указанные в нём checkpoints остаются на месте, их можно безопасно пересчитать:

```powershell
.\training_workspace\START_TRAINING.ps1 -ReevaluateRun "C:\полный\путь\nlu_report.json"
```

Этот режим не запускает optimizer и не меняет веса. Он пересчитывает
development-метрики текущим runtime-кодом, повторно применяет raw-neural и
end-to-end gates и открывает holdout только для нового устойчивого победителя.
Если старый отчёт уже содержит holdout, команда откажется повторно выбирать по
нему модель.

## 6. Проверка скорости на компьютере, где будет работать Jarvis

RTX 3090 используется для обучения, но runtime сейчас настроен на CPU. После
переноса checkpoint измерьте его на целевом компьютере:

```powershell
.\.venv-training\Scripts\python.exe -m training_workspace.benchmark_model `
  --checkpoint training_workspace/export/jarvis_nlu_best.pt --device cpu
```

Сравните с baseline:

```powershell
.\.venv-training\Scripts\python.exe -m training_workspace.benchmark_model `
  --checkpoint models/nlu_word_bigru_curriculum.pt --device cpu
```

Смотрите `intent_macro_f1`, `latency_ms_median` и `latency_ms_p95`.

## 7. Подключение к Jarvis

Сначала скопируйте победителя под новым именем — исходная модель сохранится:

```powershell
.\training_workspace\COPY_BEST_TO_MODELS.ps1
```

Затем измените в корневом `config.yaml`:

```yaml
modules:
  nlu:
    model: models/nlu_manager_finetuned.pt
```

И выполните:

```powershell
python -m pytest -v
python main.py --text "сколько сейчас времени"
python main.py --text "какие приложения ты можешь открыть"
```

Если тесты или команды стали хуже, верните путь
`models/nlu_word_bigru_curriculum.pt`. Старый checkpoint не перезаписывается.

## Что настраивать в `config.yaml`

- `search.trials`: число конфигураций первого этапа;
- `search.top_k`: сколько конфигураций проходят в дорогую проверку;
- `search.confirmation_seeds`: независимые повторы финалистов;
- `search.space`: диапазоны learning rate, warmup, cosine minimum, EMA decay,
  доли корпуса, размеров сети, веса класса `O` и весов
  route/slot/slot-consistency/no-slot losses;
- `search.phase_one_patience` и `confirmation_patience`: сколько эпох ждать
  улучшения общей development-метрики;
- `selection.min_custom_macro_f1_improvement`: прирост на новых командах;
- `selection.max_legacy_macro_f1_drop`: допустимая регрессия старых навыков;
- `selection.min_regression_worst_recall`: нижняя граница recall любого intent
  на старом regression-наборе;
- `selection.min_slot_entity_f1`: качество извлечения аргументов команды;
- `selection.max_slot_hallucination_rate`: максимум ложных slots у фраз, где
  slots не разрешены;
- `selection.min_semantic_frame_exact_match`: точная доля совпадений intent и
  всех slots одновременно;
- `selection.max_expected_calibration_error`: предел ошибки уверенности модели;
- `selection.min_holdout_worst_recall`: нижняя граница recall любого intent на
  двух финальных наборах;
- `selection.max_p95_latency_ms`: верхняя граница задержки решения.

Не используйте ни один holdout для выбора этих значений. Настройки выбираются
только по train/validation и старому regression-набору; holdout подтверждает
или отклоняет уже выбранный checkpoint.
