import itertools
import time

import mlflow
import torch

from utils.checkpoint import save_checkpoint
from utils.func_loss import calc_loss_loader, cals_loss_batch
from utils.gen import generate


def train_model(model, train_loader, val_loader,
                optimizer, device, num_epochs, eval_freq,
                eval_iter, start_context, tokenizer,
                checkpoint_path, save_freq, seed,
                resume_state=None, run_id=None):

    state = resume_state or {
        "epoch": 0, "batches_done": 0, "global_step": -1, "tokens_seen": 0,
        "train_losses": [], "val_losses": [], "track_tokens_seen": []
    }
    train_losses = state["train_losses"]
    val_losses = state["val_losses"]
    track_token_seen = state["track_tokens_seen"]
    tokens_seen, global_step = state["tokens_seen"], state["global_step"]

    epoch, batches_done = state["epoch"], state["batches_done"]

    def checkpoint(epoch, batches_done):
        save_checkpoint(checkpoint_path, model, optimizer, {
            "epoch": epoch,
            "batches_done": batches_done,
            "global_step": global_step,
            "tokens_seen": tokens_seen,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "track_tokens_seen": track_token_seen,
            "run_id": run_id
        })
        print(f"Checkpoint saved: step {global_step}")

    try:
        for epoch in range(state["epoch"], num_epochs):
            epoch_start_time = time.time()

            train_loader.generator.manual_seed(seed + epoch)

            skip = state["batches_done"] if epoch == state["epoch"] else 0
            batches_done = skip

            model.train()
            for input_batch, target_batch in itertools.islice(train_loader, skip, None):
                step_start_time = time.time()

                optimizer.zero_grad()
                loss = cals_loss_batch(
                    input_batch, target_batch, model, device
                )
                loss.backward()
                optimizer.step()
                tokens_seen += input_batch.numel()
                global_step += 1
                batches_done += 1

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

                if global_step > 0 and global_step % save_freq == 0:
                    checkpoint(epoch, batches_done)

            generate_and_print(
                model, tokenizer, device, start_context
            )

            epoch_latency = time.time() - epoch_start_time
            mlflow.log_metric(
                "epoch_latency_sec",
                epoch_latency,
                step=epoch
            )

            checkpoint(epoch + 1, 0)

    except KeyboardInterrupt:
        # Ctrl+C: сохраняемся и выходим, потом можно продолжить с --resume
        checkpoint(epoch, batches_done)
        raise

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