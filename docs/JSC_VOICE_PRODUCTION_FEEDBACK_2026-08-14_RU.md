# JSC Voice Production Feedback — 2026-08-14

Статус: живой production-feedback checkpoint по голосовым блокам A–G.

Источник истины:

- реальные Parakeet-транскрипции из session logs и переданных пользователем логов;
- параллельные решения текущего JSC shadow из `logs/jsc_shadow.jsonl`;
- фактические ответы и tool results production-контура;
- эталоны из `docs/JSC_VOICE_COLLECTION_RU.md`.

Этот документ нельзя трактовать как разрешение включить JSC в execution mode.
Текущий JSC остаётся shadow-only до выполнения safety gates ниже.

## Сводка A–G

| Блок | Основной результат |
|---|---|
| A, одиночные команды | Parakeet передаёт смысл примерно в 85% случаев. Production и JSC дают около 50% корректных итогов. Зафиксировано противоположное production-действие: команда завершить Paint открывала Paint. |
| B, одиночные системные команды | Parakeet около 91%. Семь ложных cancel-срабатываний блокировали команду до JSC shadow. Обнаружены пробелы media/router/browser families. |
| C, два действия | Exact plan: production 65%, JSC 70%. Полное фактическое выполнение только 3/15; слабые места — coreference и остановка цепочки после ошибки инструмента. |
| D, три действия | Exact plan: production и JSC 7/15 (46,7%). Полное выполнение 3/15. JSC способен обрезать план и генерировать несвязанный malformed `close`. |
| E, четыре действия | Exact plan: production и JSC 6/10 (60%). Полное выполнение 3/10. E404: вместо операций с жестами JSC предложил `window_control.close` без цели. |
| F, пять действий | Exact plan: production и JSC 4/10 (40%). Подтверждённый полностью корректный результат 1/10. F504 воспроизвёл E404 уже тремя `close` без цели. |
| G, multi-turn | Запись неполная: раздельно получены только две пары из десяти; успешных пар нет. Несмотря на это, обнаружены системные дефекты clarification state, reminder state, cancel arbitration и correction semantics. |

## G. Multi-turn: фактические наблюдения

### Покрытие записи

В логе отсутствуют вторые реплики M001–M005 и M007–M009. M010 сначала
попал в одну транскрипцию, затем был повторён двумя отдельными ходами. Поэтому
этот запуск сохраняется как failure feedback, но не используется как полный
процентный benchmark G.

Parakeet корректно сохранил смысл всех фактически захваченных эталонных реплик.
Основные ошибки G возникли после STT.

### Generic application/window clarification

Ожидаемая политика:

- `Закрой приложение` -> `ask(missing=application)`: «Какое приложение закрыть?»;
- `Нужно закрыть одно окно` -> `ask(missing=window)`: «Какое окно закрыть?»;
- `Открой приложение` -> `ask(missing=application)`: «Какое приложение открыть?»;
- `Запусти нужную программу` -> `ask(missing=application)`: «Какую программу запустить?»;
- следующая короткая реплика заполняет только запрошенный слот и продолжает
  исходное действие.

Фактически:

- trace `72e6eb47c1ea`: `Закрой приложение` превратилось в попытку закрыть окно
  с буквальным именем `приложение`;
- traces `fc883c090e43`, `115502fb59f9`, `39f2361d4737`, `43b029510ac7`:
  production отказался, а JSC не создал состояние уточнения;
- generic noun `приложение/программа/окно` нельзя принимать за конкретную цель.

### Reminder clarification and state

Ожидаемая политика:

1. Явная просьба с текстом, но без времени:
   `Напомни проверить духовку` -> «Когда вам напомнить?».
2. Явная просьба со временем, но без текста:
   `Через десять минут напомни` -> «О чём вам напомнить?».
3. Разговорное намерение:
   `Мне нужно не забыть позвонить другу`:
   - при высокой уверенности reminder intent — спросить только время;
   - при недостаточной уверенности — «Создать напоминание позвонить другу?»,
     затем запросить время после подтверждения.
4. До заполнения обязательных слотов реальное напоминание не создаётся.

Фактические дефекты:

- trace `4e1982e9c303`: корректная транскрипция `проверить духовку` была
  преобразована NLU в `верить духовку`;
- причина поддерживается кодом: optional prefix в incomplete-reminder regex
  может поглотить `про` в начале слова `проверить`; prefix должен иметь границу
  слова и обязательный пробел;
- trace `5485296aef9a`: ответ `Через десять минут` был классифицирован как новый
  `set_reminder`, из-за чего pending clarification очистился вместо merge slots;
- незавершённые `minutes=10` сохранились и trace `3c7d2d453f37` создал реальное
  напоминание №6 с текстом следующей группы M007;
- pending state не должен протекать между независимыми диалогами и должен иметь
  явные lifecycle: create, merge, complete, cancel, expire;
- JSC на trace `3c7d2d453f37` вернул `ask(missing_time)`, но поместил внутрь
  посторонний draft-step `file_control.delete`. Такой план недопустим.

### Cancel arbitration

Ложные и неверно приоритизированные cancel-срабатывания:

- `Проверить духовку` -> global cancel с confidence 0,618;
- `Создай напоминание ответить коллеге` -> global cancel с confidence 0,615;
- `Отмени напоминание` -> global cancel 1,0 вместо `cancel_reminder`.

Требования:

- domain-specific intent (`cancel_reminder`) имеет приоритет над global cancel;
- global cancel разрешён только для явных самостоятельных формулировок отмены;
- cancel gate не должен блокировать JSC shadow logging: необходимо сохранять
  кандидат JSC даже при production cancel для анализа расхождений;
- пороги сами по себе не решат ошибку: нужны negative examples и лексические
  guardrails перед применением confidence threshold.

### Correction semantics

M010 должен быть транзакцией:

1. `Открой калькулятор` выполняет или подготавливает первое действие;
2. `Нет, я имел в виду Пейнт` распознаётся как correction предыдущего target;
3. политика явно решает, нужно ли компенсировать первое действие;
4. итоговый state и ответ сообщают, что открылся Paint вместо калькулятора.

Фактически:

- trace `897fc3163251`: объединённая фраза вызвала у JSC несвязанный
  `system_control.media_play_pause`;
- раздельные traces `da765cbfcd45` и `18a4dbd98a6e`: production открыл оба
  приложения без компенсации первого, JSC на correction вернул `cancel`;
- correction нельзя сводить к независимой новой команде или общей отмене.

## Обязательная архитектура clarification

Dialogue state должен быть типизированным и журналируемым:

```text
PendingDialogue
  dialogue_id
  source_trace_id
  intent
  collected_slots
  missing_slots
  proposed_steps
  confirmation_required
  created_at
  expires_at
```

Правила:

1. `act=ask` не содержит исполняемых или посторонних steps. Допускаются только
   отдельные non-executable draft steps, которые никогда не попадут в executor.
2. Ответ пользователя сначала пытается заполнить pending slots, даже если
   одноходовой NLU распознал в нём тот же domain intent.
3. Новый явно независимый intent закрывает или приостанавливает pending state по
   детерминированной политике и пишет причину в лог.
4. Однословный ответ (`Калькулятор`, `Пейнт`) валиден только в контексте
   соответствующего pending slot.
5. History и dialogue state передаются в JSC shadow и записываются рядом с JAL.
6. Любое уточнение должно иметь понятный вопрос, а не общий отказ.

## P0 safety gates перед execution mode

- Запретить `window_control.close` без конкретного `window`.
- Запретить `file_control.delete`, если в текущем запросе и подтверждённой истории
  нет семантики удаления.
- Проверять grounding каждого tool step в глаголах и объектах запроса.
- Проверять полноту плана относительно количества независимых действий.
- Блокировать исполнение, если execute-план семантически не связан с запросом.
- Не исполнять draft steps внутри `ask/confirm/reject/cancel`.
- Исправить reminder prefix parsing и slot merge.
- Разделить global cancel и domain cancel.
- Реализовать generic application/window clarification.
- Реализовать correction state с явной compensation policy.

## Acceptance criteria для повторного G

- Все 10 групп записаны как 20 отдельных транскрипций и связаны `dialogue_id`.
- Clarification decision accuracy: не ниже 95%.
- Slot carry-over accuracy: не ниже 95%.
- End-to-end success: не ниже 90% на первом gate, затем целевые 95%.
- Ноль unrelated, destructive или targetless steps.
- Ноль ложных global cancel на эталонах G.
- Ноль переноса pending slots между независимыми группами.
- M010 оставляет систему в документированном и проверяемом конечном состоянии.

## Результат цикла fine-tuning v8

Требования этого feedback реализованы в экспериментальном production wiring:

- shadow хранит типизированный pending JAL, history и `dialogue_id`;
- generic application/window и неполные reminders возвращают `ask`;
- второй ход заполняет только ожидаемый slot;
- non-execute drafts проходят semantic grounding и не попадают в executor;
- targetless `close`, process-level запросы и отрицательные команды блокируются;
- structured data расширены до 4 355 train-примеров и обучены с category-balanced
  sampling;
- выбран checkpoint seed 29: 534 942 параметра, epoch 5;
- migration development Exact JAL: 87,75%; multi-turn: 100%; 4–5 действий:
  100%; ASR noise: 100%; false execution и opposite action: 0%.

Checkpoint подключён только в `jsc_shadow`. Production NLU остаётся
исполнителем. Удалять NLU пока нельзя: correction остаётся 46,67%, OOD exact —
33,33%, а новый независимый frozen voice holdout ещё не собран. Следующий цикл —
shadow telemetry, correction state/compensation policy и закрытый голосовой
holdout, после чего можно принимать отдельное решение об execution canary.
