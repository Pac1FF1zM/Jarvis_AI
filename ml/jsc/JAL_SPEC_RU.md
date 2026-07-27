# JAL v1 — Jarvis Action Language

JAL — компактное типизированное представление смысла пользовательской реплики.
Будущий decoder генерирует JAL, а не вызывает Python и не пишет shell-команды.
До исполнения каждый план обязан пройти этот codec, tool-schema validation и
runtime safety policy.

## Канонический формат

JAL v1 хранится как JSON UTF-8 без лишних пробелов, с отсортированными ключами.
Одинаковый план всегда имеет одинаковое строковое представление — это важно для
exact-match метрики и воспроизводимого обучения.

```json
{"act":"execute","missing":[],"reason":null,"steps":[{"arguments":{"application":"discord"},"tool":"open_application"}],"version":1}
```

Поля верхнего уровня:

- `version`: только `1`;
- `act`: `execute`, `ask`, `confirm`, `cancel`, `reject` или `dialogue`;
- `steps`: до восьми вызовов зарегистрированных инструментов;
- `missing`: ссылки `{step, name}` на отсутствующие параметры;
- `reason`: машинная причина уточнения, подтверждения, отказа или диалога.

## Acts

- `execute`: один или несколько полностью валидных шагов;
- `ask`: незавершённые шаги и непустой список `missing`;
- `confirm`: валидный план, который safety policy должен подтвердить;
- `cancel`: отмена текущего действия, без шагов;
- `reject`: неподдерживаемый/OOD запрос, с причиной;
- `dialogue`: реплика не требует инструмента; генерация ответа является
  отдельной задачей и не получает права на side effect.

## Примеры

Уточнение времени напоминания:

```json
{"act":"ask","missing":[{"name":"clock_time","step":0}],"reason":"missing_time","steps":[{"arguments":{"message":"позвонить другу"},"tool":"set_reminder"}],"version":1}
```

Составная команда:

```json
{"act":"execute","missing":[],"reason":null,"steps":[{"arguments":{"application":"discord"},"tool":"open_application"},{"arguments":{"message":"закрыть Discord","minutes":10},"tool":"set_reminder"}],"version":1}
```

## Инварианты безопасности

- неизвестные tools и arguments запрещены;
- типы, enum, обязательные и взаимоисключающие параметры берутся из реальных
  `TOOL_SCHEMA`, а не дублируются внутри модели;
- `ask` может пропустить только параметр, явно указанный в `missing`;
- строки не интерпретируются как команды shell;
- парсер отклоняет неизвестные поля, duplicate JSON keys, NaN/Infinity,
  слишком большой документ и более восьми шагов;
- валидный JAL ещё не означает разрешение side effect: allow-list и
  orchestrator остаются последним обязательным барьером.
