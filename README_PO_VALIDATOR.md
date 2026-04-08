# PO File Validator с использованием LLM (Ollama)

Production-ready CLI-скрипт для валидации больших .po файлов (65k+ записей, EN→RU) с использованием локальной LLM через Ollama.

## Возможности

- **Двухэтапная валидация:**
  - Этап 1 (Быстрый): Regex-проверка спецсимволов (`{.*}`, `%[sdf]`, `\n`, `<[^>]+>`), подсчёт предложений и строк
  - Этап 2 (Семантика): Отправка пары msgid/msgstr в локальную Ollama для проверки смысла

- **Устойчивость к сбоям:**
  - Чекпоинты каждые 100 записей в `resume_state.json`
  - Graceful shutdown при Ctrl+C с сохранением прогресса
  - Ретраи: 3 попытки с экспоненциальной задержкой (1s, 2s, 4s)
  - Валидация JSON от LLM с обработкой ошибок парсинга

- **Гибкое управление:**
  - Лимит на количество обрабатываемых записей (`--limit`)
  - Возобновление с места остановки (`--resume`)
  - Изменение лимита при возобновлении

## Установка зависимостей

```bash
pip install polib requests tqdm
```

## Быстрый старт

### Полная валидация файла

```bash
python po_validator.py input.po --model gemma3:4b
```

### Валидация с ограничением по количеству записей

```bash
# Проверить только первые 1000 записей
python po_validator.py input.po --model gemma3:4b --limit 1000

# Проверить следующие 500 записей (возобновление с новым лимитом)
python po_validator.py input.po --model gemma3:4b --limit 1500 --resume
```

### Возобновление прерванной валидации

```bash
# Продолжить с места остановки
python po_validator.py input.po --model gemma3:4b --resume

# Продолжить с новым лимитом
python po_validator.py input.po --model gemma3:4b --limit 5000 --resume
```

### Кастомные параметры вывода

```bash
python po_validator.py input.po \
    --model gemma3:4b \
    --output validated_output.po \
    --issues issues.jsonl \
    --timeout 60 \
    --verbose
```

## Все команды и опции

### Базовые команды

| Команда | Описание |
|---------|----------|
| `python po_validator.py input.po` | Запуск полной валидации |
| `python po_validator.py input.po --resume` | Возобновление прерванной валидации |
| `python po_validator.py input.po --limit N` | Проверить только N записей |

### Параметры командной строки

| Параметр | Короткая форма | Описание | Пример |
|----------|----------------|----------|--------|
| `input` | - | Путь к входному .po файлу (обязательно) | `ru.po` |
| `--output` | `-o` | Путь к выходному .po файлу | `--output validated.po` |
| `--issues` | - | Путь к JSONL файлу с ошибками | `--issues report.jsonl` |
| `--resume-state` | - | Путь к файлу состояния | `--resume-state checkpoint.json` |
| `--resume` | - | Возобновить с последнего чекпоинта | `--resume` |
| `--limit` | - | Лимит записей для обработки | `--limit 1000` |
| `--model` | - | Модель Ollama | `--model gemma3:4b` |
| `--timeout` | - | Таймаут запроса в секундах | `--timeout 60` |
| `--verbose` | `-v` | Включить подробный вывод | `-v` |

## Сценарии использования

### Сценарий 1: Тестирование на малой выборке

```bash
# Проверить первые 100 записей для теста
python po_validator.py ru.po --model gemma3:4b --limit 100
```

### Сценарий 2: Пошаговая обработка большого файла

```bash
# Шаг 1: Первые 1000 записей
python po_validator.py ru.po --model gemma3:4b --limit 1000

# Шаг 2: Следующие 1000 (всего 2000)
python po_validator.py ru.po --model gemma3:4b --limit 2000 --resume

# Шаг 3: Продолжить до конца
python po_validator.py ru.po --model gemma3:4b --resume
```

### Сценарий 3: Обработка с перерывами

```bash
# Запустить обработку
python po_validator.py ru.po --model gemma3:4b

# Нажать Ctrl+C для остановки (прогресс сохранится)

# Позже возобновить
python po_validator.py ru.po --model gemma3:4b --resume
```

### Сценарий 4: Использование другой модели

```bash
# Список доступных моделей
curl http://localhost:11434/api/tags

# Использовать конкретную модель
python po_validator.py ru.po --model qwen2.5:7b --timeout 45
```

## Выходные файлы

### validated_output.po
Копия оригинального .po файла с добавленными флагами `#freez` перед проблемными записями:

```po
#freez
#: src/file.py:42
msgid "Hello"
msgstr "Как дела?"
```

### issues.jsonl
JSON Lines формат с информацией об ошибках. Каждая строка — отдельный объект:

```json
{"index": 42, "msgid": "Hello", "msgstr": "Как дела?", "reason": "semantic_mismatch", "type": "semantic"}
{"index": 157, "msgid": "Value: {name}", "msgstr": "Значение:", "reason": "placeholder_mismatch: missing in target: ['{name}']", "type": "structural"}
```

### resume_state.json
Файл состояния для возобновления (автоматически удаляется после успешного завершения):

```json
{
  "last_index": 1234,
  "issues": [...],
  "processed_count": 1235,
  "error_count": 42,
  "total_to_process": 5000
}
```

## Типы обнаруживаемых ошибок

### Структурные ошибки (Stage 1)
- **placeholder_mismatch**: Несовпадение плейсхолдеров (`{name}`, `%s`, `\n`, HTML-теги)
- **sentence_count_mismatch**: Разное количество предложений (допускается ±1)
- **line_count_mismatch**: Разное количество строк

### Семантические ошибки (Stage 2)
- **semantic_mismatch**: Смысловой разрыв (определяется LLM)
- **llm_parse_error**: Ошибка парсинга JSON от LLM
- **ollama_timeout**: Таймаут запроса к Ollama
- **ollama_connection_error**: Ошибка подключения к Ollama

## Рекомендации по производительности

### Для RTX 3050 (4GB VRAM)
- Используйте легковесные модели: `gemma3:4b`, `qwen2.5:7b`, `llama3.2:3b-instruct-q4_k_m`
- Увеличьте таймаут для больших моделей: `--timeout 60`
- Обрабатывайте порциями с `--limit`

### Оптимизация скорости
1. Закройте другие приложения, использующие GPU
2. Используйте квантованные модели (q4_k_m)
3. Запустите Ollama в фоновом режиме перед валидацией

## Требования к системе

- Python 3.10+
- Ollama (локально запущенный)
- Библиотеки: `polib`, `requests`, `tqdm`

## Запуск Ollama

```bash
# Установка (если не установлен)
# См. https://ollama.ai

# Запуск сервера
ollama serve

# В другом терминале - загрузка модели
ollama pull gemma3:4b

# Проверка доступных моделей
ollama list
```

## Примеры вывода

### Успешное завершение
```
============================================================
VALIDATION COMPLETE
Total entries processed: 64737
Issues found: 1523
  - Structural: 847
  - Semantic: 676
Output files:
  - Validated PO: validated_output.po
  - Issues JSONL: issues.jsonl
============================================================
```

### Прерывание с сохранением прогресса
```
^CInterrupt signal received. Will save checkpoint after current entry...
Checkpoint saved at index 1234. Stopping gracefully.
Graceful stop completed. Use --resume to continue.
```

## Устранение неполадок

### Ошибка: "Cannot connect to Ollama"
```bash
# Проверьте, запущен ли Ollama
ollama serve

# Проверьте порт по умолчанию
curl http://localhost:11434/api/tags
```

### Ошибка: "Model not found"
```bash
# Скачайте нужную модель
ollama pull gemma3:4b

# Или используйте доступную
python po_validator.py input.po --model llama3.2:3b-instruct-q4_k_m
```

### Медленная обработка
- Уменьшите размер модели
- Увеличьте `--timeout` для избежания ретраев
- Используйте `--limit` для обработки порциями

## Лицензия

MIT License
