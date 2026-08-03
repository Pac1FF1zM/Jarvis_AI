# Автоматическая запись пользовательского датасета

Программа показывает текущий класс, делает обратный отсчёт, записывает трёхсекундный клип, автоматически присваивает метку и добавляет строку в `data/custom_capture/manifest.csv`.

## Порядок записи

Запускайте серии в разные моменты времени. Между сериями желательно немного изменить освещение, положение камеры или расстояние до неё.

### 1. Обучающая серия

```powershell
.\.venv-training\Scripts\python.exe -m src.record_custom_dataset --split train
```

Будет записано 80 клипов: по 5 для каждого жеста и 15 примеров `D0X`.

### 2. Validation-серия

```powershell
.\.venv-training\Scripts\python.exe -m src.record_custom_dataset --split val
```

Будет записан 31 клип: по 2 для каждого жеста и 5 примеров `D0X`.

### 3. Финальная тестовая серия

```powershell
.\.venv-training\Scripts\python.exe -m src.record_custom_dataset --split test
```

Будет записан 31 клип. Не используйте эту серию для настройки модели до фиксации финального checkpoint.

## Управление

- `Space` — запустить серию после открытия окна камеры.
- `R` — перезаписать только что снятый клип во время экрана проверки.
- `Enter` — принять клип сразу, не ожидая автоматического принятия.
- `Esc` — безопасно завершить текущую серию. Уже принятые клипы сохранятся.

Превью не отражается по горизонтали: в файл сохраняется ровно то направление движения, которое видно в окне.

Если основная камера имеет другой индекс, добавьте, например, `--camera 1`.

```powershell
.\.venv-training\Scripts\python.exe -m src.record_custom_dataset --split train --camera 1
```

Для короткой проверки камеры можно записать по одному клипу каждого жеста и два `D0X`:

```powershell
.\.venv-training\Scripts\python.exe -m src.record_custom_dataset --split train --gesture-repetitions 1 --non-gesture-repetitions 2
```

Проверка плана без открытия камеры:

```powershell
.\.venv-training\Scripts\python.exe -m src.record_custom_dataset --split train --dry-run
```
