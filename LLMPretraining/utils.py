import torch
from llm_gpt2 import GPT
import torch.distributed as dist
import time


def remove_prefix(state_dict, prefix):
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def load_model(check_point_file):
    checkpoint = torch.load(check_point_file)  # read checkpoint from log
    model = GPT(checkpoint["config"])  # GPT2 model with rand init weights
    state_dict = remove_prefix(checkpoint["model"], "_orig_mod.")
    model.load_state_dict(state_dict)  # load weights from checkpoint
    return model


def eval_gpt2(
    model, eval_loader, ddp=True, total_eval_steps=10, device="cuda", device_type="cuda"
):
    if ddp:
        eval_loader.reset()
    model.eval()
    eval_loss_accum = 0.0
    t0 = time.time()
    with torch.no_grad():
        for i in range(total_eval_steps):
            x, y = eval_loader.next_batch()
            x, y = x.to(device=device), y.to(device=device)

            # trick 2: Auto Mixed Precison (https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
            # logits: bfloat16, weight, loss: float32
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss = loss / total_eval_steps
            eval_loss_accum += loss.detach()
    if ddp:  # AVG the loss across all GPUs
        dist.all_reduce(eval_loss_accum, dist.ReduceOp.AVG)
    t1 = time.time()
    dt = (t1 - t0) * 1000
    return eval_loss_accum.item()
