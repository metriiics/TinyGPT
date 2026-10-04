import json

import torch
from torch.utils.data import DataLoader
import torch.optim as op
from tokenizers import Tokenizer

from model.config import GPTConfig
from utils.data import GPTDataset, BinDataset
from utils.train import train_model
from utils.plotting import plot_losses
from model.gpt import GPTModel


device = torch.device("cuda")

tokenizer = Tokenizer.from_file("Tokenizer/vocabulary/tokenizer.json")

with open("parameters.json", "r", encoding="utf-8") as file:
    cfg = json.load(file)

conf = GPTConfig(**cfg["TINY_GPT_CONFIG"])

batch_size = 64

train_data = BinDataset(
    r"datasets\processed\train.bin",
    max_length=conf.context_length,
    stride=conf.context_length
)
val_data = BinDataset(
    r"datasets\processed\val.bin",
    max_length=conf.context_length,
    stride=conf.context_length
)


train_dl = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0
)
valid_dl = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=0
)

model = GPTModel(conf)
model.to(device)

optim = op.AdamW(
    model.parameters(),
    lr=0.0004, weight_decay=0.1
)

num_epochs = 50
train_losses, val_losses, tokens_seen = train_model(
    model, train_dl, valid_dl, optim, device, 
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Ма́монты () — вымерший род", tokenizer=tokenizer
)

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict()
    },
    "weight/model_and_optimizer.pth"
)