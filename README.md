# Listing Guard

Модерация объявлений маркетплейса Neon готовыми ML-моделями: текст (toxic-bert), голос (Whisper), фото (CLIP),
видео (YOLO11s с размытием людей) и заполнение карточки товара через Qwen3-VL-8B в vLLM.
Подробное описание моделей и демо - в `demo.ipynb`.

## Требования

- macOS 15+ на Apple Silicon (vLLM работает через плагин vllm-metal на MLX)
- [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- ffmpeg: `brew install ffmpeg`
- около 10 ГБ свободного места под модели

## 1. Установка проекта

```bash
uv sync
uv run python scripts/download_data.py
```

`uv sync` ставит зависимости вместе с dev-группой (Jupyter, matplotlib).
Скрипт скачивает демо-данные в `data/`: фото товаров, фото с запрещённым предметом и видео.

## 2. Запуск vLLM

vLLM ставится в отдельное окружение, чтобы его зависимости не конфликтовали с проектом.
Установщик vllm-metal сам создаёт окружение `~/.venv-vllm-metal` через uv:

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

Запуск сервера в отдельном терминале:

```bash
source ~/.venv-vllm-metal/bin/activate
vllm serve mlx-community/Qwen3-VL-8B-Instruct-4bit \
    --served-model-name qwen3-vl-8b \
    --max-model-len 16384 \
    --port 8000
```

При первом запуске модель скачивается с Hugging Face. Сервер готов, когда отвечает:

```bash
curl http://localhost:8000/v1/models
```

Проект по умолчанию ходит в `http://localhost:8000/v1` к модели `qwen3-vl-8b`.
Другой адрес или модель задаются переменными `LLM_BASE_URL` и `LLM_MODEL`.

## 3. Демо в Jupyter

```bash
uv run jupyter lab demo.ipynb
```

Выполнение ячеек по порядку. Первая ячейка генерирует голосовые командой `say` (macOS)
и прогревает модели, при первом запуске она скачивает веса toxic-bert, Whisper, CLIP и YOLO.
Если vLLM не запущен, модерация работает, но карточка товара не заполняется.
