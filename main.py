import argparse
import json
import os

import mlflow
import torch
import torch.optim as op
from dotenv import load_dotenv
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from model.config import GPTConfig
from model.gpt import GPTModel
from utils.checkpoint import load_checkpoint
from utils.data import BinDataset
from utils.plotting import plot_losses
from utils.train import train_model

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true", help="продолжить с чекпойнта")
args = parser.parse_args()

CHECKPOINT = "weight/checkpoint.pth"
SEED = 123
os.makedirs("weight", exist_ok=True)


mlflow.set_tracking_uri(os.getenv("MLFLOW_URI"))
mlflow.set_experiment("gpt-tiny")

device = torch.device("cuda")

tokenizer = Tokenizer.from_file("Tokenizer/vocabulary/tokenizer.json")

with open("parameters.json", "r", encoding="utf-8") as file:
    cfg = json.load(file)

conf = GPTConfig(**cfg["TINY_GPT_CONFIG"])

batch_size = 32
num_epochs = 2
lr = 0.0004
weight_decay=0.1

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
    num_workers=0,
    generator=torch.Generator()
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
    lr=lr, weight_decay=weight_decay
)

resume_state, run_id = None, None
if args.resume:
    resume_state = load_checkpoint(CHECKPOINT, model, optim, device)
    run_id = resume_state["run_id"]
    print(f"Resumed from step {resume_state['global_step']}")

run_name = f"tiny-gpt_lr{lr}_bs{batch_size}_ep{num_epochs}"

run_args = {"run_id": run_id} if run_id else {"run_name": run_name}

with mlflow.start_run(**run_args, log_system_metrics=True) as run:

    if run_id is None:
        mlflow.log_params(
            {
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "optimizer": "AdamW"
            }
        )

        mlflow.log_params({f"config_{k}": v for k, v in conf.__dict__.items()
            if isinstance(v, (int, float, str, bool))})

        mlflow.set_tags(
            {
                "model": "GPT-Tiny",
                "dataset": "PotatoHD/ru-text-corpus",
                "device": str(device)
            }
        )

    train_losses, val_losses, tokens_seen = train_model(
        model, train_dl, valid_dl, optim, device,
        num_epochs=num_epochs, eval_freq=500, eval_iter=50,
        start_context="Ма́монты () — вымерший род", tokenizer=tokenizer,
        checkpoint_path=CHECKPOINT, save_freq=10000, seed=SEED,
        resume_state=resume_state, run_id=run.info.run_id
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict()
        },
        "weight/model_and_optimizer.pth"
    )

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)