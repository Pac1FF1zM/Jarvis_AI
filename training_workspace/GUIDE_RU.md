# Fine-tuning собственной NLU-модели Jarvis на RTX 3090

Этот workspace дообучает только собственную `Word-BiGRU` Jarvis. Он не
загружает модели, токенизаторы или датасеты Hugging Face. Исходный checkpoint
был обучен с нуля, а PyTorch используется как вычислительный фреймворк.

## Что делает один запуск

`START_TRAINING.ps1` последовательно:

1. проверяет JSONL, отсутствие пересечения custom train/validation и CUDA;
2. измеряет accuracy, macro-F1 и latency исходной модели;
3. запускает три режима fine-tuning: `augmented`, `curriculum`, `standard`;
4. для каждого режима загружает старые веса, расширяет vocabulary новыми
   токенами и сохраняет старые token IDs/embedding rows;
5. смешивает custom-примеры с базовым корпусом, чтобы не забыть старые команды;
6. использует AMP, TF32, balanced sampling, gradient clipping, label smoothing
   и early stopping по validation macro-F1;
7. сравнивает кандидатов на одинаковом validation-наборе и CPU latency;
8. экспортирует модель только если она не хуже baseline и укладывается в SLA;
9. после выбора победителя один раз проверяет `evaluation_holdout.jsonl`.

RTX 3090 ускорит серию экспериментов, но NLU содержит меньше 100 тысяч
параметров. Главный источник роста точности — разнообразные и правильно
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

В репозитории уже подготовлен сбалансированный датасет v2:

- `train.jsonl`: 840 примеров, по 120 на каждый intent;
- `validation.jsonl`: 210 примеров, по 30 на каждый intent;
- `evaluation_holdout.jsonl`: 105 примеров, по 15 на каждый intent.

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

## 3. Проверка до долгого запуска

```powershell
.\training_workspace\START_TRAINING.ps1 -CheckOnly
```

Команда проверит пути, JSON, intents, slots и статистику классов без обучения.

## 4. Запуск

Перенесите на компьютер друга весь актуальный проект. Для обучения обязательно
нужны как минимум:

- `models/nlu_word_bigru_curriculum.pt` — проверенный базовый checkpoint;
- весь каталог `training_workspace`, включая новый `data/`;
- каталоги `ml/nlu` и корневые Python-зависимости проекта.

Не используйте старый `training_workspace/export/jarvis_nlu_best.pt`, который
был получен на демонстрационном наборе из 14 train-примеров, как новую основу.
Конфигурация уже правильно начинает fine-tuning от базового checkpoint, а
после экспериментов заменяет файл в `export/` только прошедшим отбор победителем.

```powershell
.\training_workspace\START_TRAINING.ps1
```

Настройки экспериментов находятся в `config.yaml`. На RTX 3090 можно сначала
оставить `batch_size: 256`. Если возникает CUDA OOM, уменьшайте до 128/64; если
GPU почти пуст и данных стало много — увеличивайте до 512.

Не закрывайте PowerShell. Каждая эпоха печатает loss, validation macro-F1 и
slot F1. Early stopping завершает бесполезные эпохи автоматически.

## 5. Результаты

Каждый запуск создаёт:

```text
training_workspace/runs/YYYYMMDD_HHMMSS/
  conservative_augmented.pt
  balanced_curriculum.pt
  focused_standard.pt
  *.metrics.json
  report.json
```

Если кандидат прошёл accuracy/latency-ограничения, победитель появится здесь:

```text
training_workspace/export/jarvis_nlu_best.pt
```

Если export не появился, это нормальная защита: ни одна модель не превзошла
baseline. Изучите `report.json`, улучшите данные и запустите новый цикл; не
понижайте пороги только ради появления файла.

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
    model: models/nlu_word_bigru_finetuned.pt
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

- `experiments[].learning_rate`: обычно `0.0002–0.001` для fine-tuning;
- `custom_repeat`: влияние новых примеров; слишком большое значение вызывает
  forgetting/переобучение;
- `label_smoothing`: снижает чрезмерную уверенность;
- `patience`: сколько эпох ждать улучшения;
- `selection.min_macro_f1_improvement`: минимальный прирост над baseline;
- `selection.max_p95_latency_ms`: верхняя граница задержки решения.

Не используйте evaluation holdout для выбора этих значений.
