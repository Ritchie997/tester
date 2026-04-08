# РОЛЬ
Ты — Senior Python Developer & AI Integration Engineer. Твоя задача — написать production-ready CLI-скрипт для валидации больших `.po` файлов (65k+ записей, EN→RU) с использованием локальной LLM через Ollama.

# КОНТЕКСТ И ЦЕЛЬ
- Вход: `.po` файл с единичными и множественными формами (`msgstr[0]`, `msgstr[1]`).
- Задача: Выявить записи с некорректным переводом (смысловой разрыв, напр. "hello" → "как дела?"), нарушенным количеством предложений/строк, потерей спецсимволов/плейсхолдеров.
- Результат: Пометить проблемные записи кастомным флагом `#freez` в копии `.po` файла И выгрузить их в отдельный отчёт для ручной проверки. AI-перевод НЕ генерировать.
- Аппаратные ограничения: RTX 3050, Ollama. Обработка строго последовательная. Приоритет: стабильность + макс. скорость без перегруза VRAM.

# ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ
1. Язык: Python 3.10+
2. Библиотеки: `polib`, `requests`, `re`, `json`, `logging`, `tqdm`, `argparse`. Без тяжёлых ML-фреймворков.
3. Архитектура валидации:
   - ЭТАП 1 (Быстрый, код): Regex-проверка спецсимволов (`{.*}`, `%[sdf]`, `\\n`, `<[^>]+>`) и подсчёт предложений/строк. Сравнивай множества и длины. При несовпадении → сразу помечать `#freez`, пропускать LLM.
   - ЭТАП 2 (Семантика, LLM): Если структура ок, отправлять пару `msgid`/`msgstr` в локальную Ollama. Каждый запрос строго изолирован. Контекст НЕ накапливается.
4. Устойчивость и чекпоинты:
   - Сохранять прогресс в `resume_state.json` (индекс последней записи, список найденных ошибок). При перезапуске с флагом `--resume` продолжать с места обрыва.
   - Ретраи: 3 попытки при ошибке сети/таймаута с `time.sleep(1, 2, 4)`.
   - Валидация JSON от LLM: если парсинг падает → считать запись ошибкой, записать `reason: "llm_parse_error"`, пометить `#freez`.
5. Вывод:
   - `validated_output.po`: копия оригинала с добавленными `#freez` перед проблемными записями.
   - `issues.jsonl`: JSON Lines формат. Каждая строка = отдельный объект `{"index": int, "msgid": str, "msgstr": str, "reason": str, "type": "semantic"|"structural"}`.
   - Консоль: прогресс-бар `tqdm`, скорость (зап/мин), счётчики ошибок/пропусков.

# ШАБЛОН ПРОМТА ДЛЯ LLM (ВСТРОИТЬ В КОД КАК КОНСТАНТУ)
```json
{
  "system": "You are a strict translation QA validator. You analyze English-to-Russian pairs. Output ONLY valid JSON. No explanations, no markdown, no code blocks.",
  "user_template": "Check semantic correctness of this translation pair.\nSOURCE: {msgid}\nTARGET: {msgstr}\n\nRules:\n1. Does TARGET convey the EXACT same meaning as SOURCE? (Flag obvious mismatches like 'hello' → 'как дела?')\n2. Ignore minor stylistic differences. Focus on meaning, context, and key terms.\n3. Return ONLY this JSON format:\n{\"is_correct\": boolean, \"reason\": \"string or null\"}\nIf correct: {\"is_correct\": true, \"reason\": null}\nIf incorrect: {\"is_correct\": false, \"reason\": \"brief 1-sentence explanation of mismatch\"}"
}
