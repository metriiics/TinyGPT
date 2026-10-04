import time

import torch
import mlflow

from utils.func_loss import cals_loss_batch, calc_loss_loader
from utils.gen import generate


def train_model(model, train_loader, val_loader,
                        optimizer, device, num_epochs, eval_freq, 
                        eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_token_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        model.train()
        for input_batch, target_batch in train_loader:
            step_start_time = time.time()

            optimizer.zero_grad()
            loss = cals_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            mlflow.log_metric(
                "train_step_loss", 
                loss.item(), 
                step=global_step
            )

            if global_step % 10 == 0:
                step_latency = time.time() - step_start_time
                mlflow.log_metric("step_latency_sec", step_latency, step=global_step)

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(tokens_seen)

                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "tokens_seen": tokens_seen,
                        "epoch": epoch
                    },
                    step=global_step
                )

                step_start_time = time.time()

                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print(
            model, tokenizer, device, start_context
        )

        epoch_latency = time.time() - epoch_start_time
        mlflow.log_metric(
            "epoch_latency_sec",
            epoch_latency,
            step=epoch
        )
        epoch_start_time = time.time()

    return train_losses, val_losses, track_token_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]

    encoded = tokenizer.encode(start_context)
    encoded_ids = encoded.ids
    encoded_tensor = torch.tensor(encoded_ids).unsqueeze(0).to(device)

    with torch.no_grad():
        token_ids = generate(
            model=model, idx=encoded_tensor,
            max_new_tokens=50, context_size=context_size
        )

    flat = token_ids.squeeze(0)
    decoded_text = tokenizer.decode(flat.tolist())

    print(decoded_text.replace("\n", " "))
    model.train()