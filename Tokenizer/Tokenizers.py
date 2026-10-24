import os

from dotenv import load_dotenv
from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers import pre_tokenizers as pre_tok
from tokenizers.decoders import ByteLevel
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

load_dotenv()

ds = load_dataset(
    "PotatoHD/ru-text-corpus", 
    token=os.getenv("HF_TOKEN")
)

split = ds["train"].train_test_split(
    test_size=0.005, 
    seed=123,
    shuffle=True
)
train = split["train"]

sample = train.shuffle(seed=0).select(
    range(
        min(300_000, len(train))
    )
)


def batch_iterator(batch_size=1000):
    for i in range(0, len(sample), batch_size):
        yield sample[i:i + batch_size]["text"]


tokenizer = Tokenizer(BPE())

trainer = BpeTrainer(
    vocab_size=21823, 
    special_tokens=["<unk>", "<pad>", "<eos>"],
    min_frequency=2
)

tokenizer.pre_tokenizer = pre_tok.ByteLevel()
tokenizer.decoder = ByteLevel()

tokenizer.train_from_iterator(
    batch_iterator(), 
    trainer=trainer,
    length=len(sample)
)

tokenizer.save("Tokenizer/vocabulary/tokenizer.json")