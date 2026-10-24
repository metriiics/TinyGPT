# TinyGPT

Небольшая GPT-подобная языковая модель (decoder-only трансформер), написанная с нуля на PyTorch и обучаемая на русскоязычном корпусе [PotatoHD/ru-text-corpus](https://huggingface.co/datasets/PotatoHD/ru-text-corpus). Эксперименты логируются в MLflow.

## Возможности

- Собственная реализация GPT: masked multi-head attention, pre-LayerNorm decoder-блоки, FFN с GELU, обучаемые позиционные эмбеддинги.
- Byte-level BPE-токенизатор (Hugging Face `tokenizers`), обученный на train-части того же корпуса.
- Данные хранятся в бинарном формате `uint16` и читаются через `np.memmap`, поэтому корпус не загружается в память целиком.
- Логирование параметров, лоссов, латентности и системных метрик в MLflow.
- Генерация текста после каждой эпохи (greedy / temperature / top-k).

## Структура проекта

```
tinygpt/
├── main.py                     # точка входа: обучение модели
├── parameters.json             # конфигурации моделей (TINY / SMALL)
├── model/
│   ├── config.py               # GPTConfig (dataclass)
│   ├── attention.py            # MaskedMultiHeadAttention
│   ├── ffn.py                  # FeedForwardNetwork
│   ├── decoder_block.py        # TransformerDecoderBlock
│   └── gpt.py                  # GPTModel
├── utils/
│   ├── data.py                 # BinDataset (memmap), GPTDataset (из сырого текста)
│   ├── func_loss.py            # cross-entropy на батче и по загрузчику
│   ├── train.py                # цикл обучения, оценка, MLflow
│   ├── gen.py                  # генерация текста
│   └── plotting.py             # графики train/val loss
├── Tokenizer/
│   ├── Tokenizers.py           # обучение BPE-токенизатора
│   └── vocabulary/tokenizer.json
├── datasets/                   # не хранится в git
│   ├── ru_text_corpus/prepare.py
│   └── processed/              # train.bin, val.bin
└── weight/                     # чекпойнты, не хранится в git
```

## Установка

Нужны Python 3.13, [uv](https://docs.astral.sh/uv/) и видеокарта NVIDIA с CUDA. PyTorch ставится из индекса `cu128`.

```bash
uv sync
```

Создайте в корне проекта файл `.env`:

```env
HF_TOKEN=hf_...                       # токен Hugging Face для загрузки датасета
MLFLOW_URI=http://127.0.0.1:5000      # адрес MLflow tracking-сервера
```

## Пайплайн

Все скрипты запускаются **из корня проекта**: пути в них относительные.

### 1. Обучение токенизатора

```bash
uv run Tokenizer/Tokenizers.py
```

Скрипт загружает ru-text-corpus, повторяет то же разбиение train/val, что и `prepare.py` (`test_size=0.005`, `seed=123`), и обучает BPE на случайной выборке из 300 000 документов **только из train-части**. Результат: `Tokenizer/vocabulary/tokenizer.json`.

- Размер словаря: 21 823 токена.
- Специальные токены: `<unk>` = 0, `<pad>` = 1, `<eos>` = 2.

### 2. Подготовка данных

```bash
uv run datasets/ru_text_corpus/prepare.py
```

Скрипт токенизирует корпус, добавляет `<eos>` в конец каждого документа и записывает все id подряд в `datasets/processed/train.bin` и `val.bin` (`uint16`, 2 байта на токен).

> После каждого переобучения токенизатора этот шаг нужно выполнить заново: `.bin` хранят id, а они зависят от словаря.

Посмотреть содержимое `.bin`:

```python
import numpy as np
from tokenizers import Tokenizer

data = np.memmap("datasets/processed/train.bin", dtype=np.uint16, mode="r")
tokenizer = Tokenizer.from_file("Tokenizer/vocabulary/tokenizer.json")

print(len(data))                              # число токенов
print(tokenizer.decode(data[:300].tolist()))  # начало корпуса
```

### 3. MLflow

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Интерфейс откроется по адресу http://127.0.0.1:5000, эксперимент называется `gpt-tiny`.

### 4. Обучение

```bash
uv run main.py
```

Модель обучается с конфигурацией `TINY_GPT_CONFIG`. После каждой эпохи она генерирует продолжение стартовой фразы. В конце показывается график лоссов, а веса модели и состояние оптимизатора сохраняются в `weight/model_and_optimizer.pth`.

## Конфигурации модели

Задаются в [parameters.json](parameters.json):

| Параметр | TINY | SMALL |
|---|---|---|
| `vocab_size` | 21 823 | 21 823 |
| `context_length` | 256 | 256 |
| `emb_dim` | 256 | 768 |
| `n_heads` | 4 | 12 |
| `n_layers` | 4 | 12 |
| `drop_rate` | 0.1 | 0.1 |
| `qkv_bias` | false | false |
| Параметров | ~14.4M | — |

`vocab_size` должен совпадать с размером словаря токенизатора.

Гиперпараметры обучения задаются в [main.py](main.py):

| Параметр | Значение |
|---|---|
| Оптимизатор | AdamW |
| `lr` | 4e-4 |
| `weight_decay` | 0.1 |
| `batch_size` | 64 |
| `num_epochs` | 2 |

## Данные

| Сплит | Токенов | Размер |
|---|---|---|
| train | ~611M | 1.2 ГБ |
| val | ~3.1M | 6 МБ |

Около 794 тыс. документов, в среднем ~770 токенов на документ.

## Лицензия

[MIT](LICENSE)
