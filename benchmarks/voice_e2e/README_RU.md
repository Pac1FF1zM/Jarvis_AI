# Независимый voice E2E benchmark

Этот набор намеренно не содержит синтетическое аудио. Для допуска JSC к
исполнению нужен замороженный JSONL-манифест минимум из 30 записей трёх или
более людей, не участвовавших в разработке. Каждая строка содержит `id`,
`path`, `audio_sha256`, человеческий `reference_text`, канонический
`expected_jal`, `speaker_id`, `provenance: "independent_human"` и
`consent: true`.

Запуск:

```powershell
python training_workspace/run_voice_e2e_benchmark.py `
  --manifest benchmarks/voice_e2e/private/manifest.jsonl `
  --output reports/voice_e2e_seed29.json
```

Runner не вызывает инструменты и отдельно измеряет качество STT, semantic plan
на эталонном тексте, полный audio→STT→JSC путь, false execution, полноту плана
и задержку. Каталог `private/` должен оставаться вне Git: в нём персональные
голосовые данные и согласия участников.
