# Обучение Jarvis Gesture Core на IPN Hand

Это отдельный ML-контур для распознавания жестов рукой через веб-камеру.
Он использует только RGB-видео IPN Hand и нейросети, созданные в этом
репозитории с нулевой инициализацией. Нет Hugging Face, MediaPipe, готовых
весов, embeddings или скачиваемого backbone.

## Что скачать

С официальной страницы IPN Hand скачайте только:

- **MP4 videos** — пять файлов `videos01.tgz`…`videos05.tgz`, всего 4.6 ГБ;
- **Annotations** — небольшой архив с JSON-разметкой.

Не нужны `Video frames`, `Optical flow frames` и `Hand segmentation frames`.
Не извлекайте MP4 в JPEG: набор из кадров заметно больше исходных видео.

IPN Hand содержит 13 жестов и `D0X` (обычные движения без команды). Этот
последний класс критичен: он измеряет ложные срабатывания, а не только
красивую accuracy на жестах.

Источник и лицензия: <https://gibranbenitez.github.io/IPN_Hand/> (CC BY 4.0).

## Подготовка на RTX 3090 Ti

Перенесите **весь актуальный проект Jarvis** на компьютер друга, затем в
PowerShell из корня проекта выполните один раз:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\training_workspace\SETUP_RTX3090.ps1 -PythonVersion "-3.12" -CudaWheel "cu128"
.\.venv-training\Scripts\python.exe -m pip install opencv-python-headless
```

Последняя команда добавляет только декодер локальных MP4. Она не скачивает
никаких моделей.

## Распаковка и импорт

Создайте, например, папку `D:\JarvisGesture\ipn\videos`. Распакуйте туда
каждый `.tgz`, чтобы внутри в итоге лежали `.mp4`. Разметку распакуйте в
`D:\JarvisGesture\ipn\annotations`.

Создайте проверенный manifest:

```powershell
.\training_workspace\IMPORT_IPN_HAND.ps1 `
  -Videos "D:\JarvisGesture\ipn\videos" `
  -Annotations "D:\JarvisGesture\ipn\annotations"
```

Скрипт найдёт официальный ActivityNet-style JSON, сопоставит все сегменты с
MP4 и откажется создавать manifest при неизвестной метке или отсутствующем
видео. В `training_workspace/gesture_data/` появится только маленький JSONL;
сами видео в Git не добавляются.

## Проверка до обучения

```powershell
.\training_workspace\START_GESTURE_TRAINING.ps1 -CheckOnly
```

Ожидается JSON с CUDA, количеством сегментов, количеством отдельных видео в
`train` / `validation` / `test` и распределением меток. Если тут ошибка —
обучение запускать не надо: сначала исправляется путь или распаковка.

## Обучение

```powershell
.\training_workspace\START_GESTURE_TRAINING.ps1
```

Будут обучены три разные модели с нуля:

1. `tiny_3d_cnn` — сразу учит пространственно-временной паттерн;
2. `cnn_bigru` — свёрточный анализ каждого кадра + двунаправленный GRU для
   траектории движения;
3. `cnn_temporal_transformer` — свёрточный анализ кадров + temporal attention.

Сравнение архитектур выполняется только по `validation`. Официальный `test`
раскрывается один раз после выбора победителя. В результатах важны не только
`macro_f1`, но и `no_gesture_recall` и `false_trigger_rate`.

Веса экспортируются в `training_workspace/gesture_export/` только если
выбранная модель проходит все safety-gates в `gesture_config.yaml`. Если
экспорта нет — это честный результат: модель ещё не готова управлять Jarvis.

## После baseline

IPN не включает жесты именно Jarvis вроде «ладонь = стоп». Их добавим вторым
этапом небольшой собственной выборкой; она должна быть отделена по людям и
освещению от финального теста. Пока нельзя подмешивать тест IPN или его видео
в тренировку ради улучшения метрики.

## Важная граница

На этом этапе создаются и проверяются **веса классификатора**. Он ещё не
подключён к управлению Windows или камере Jarvis: выполнение действий появится
только после отдельного модуля runtime с режимом активации, порогом уверенности
и подтверждением нескольких последовательных кадров. Одна классификация не
должна открывать приложения или нажимать клавиши.
