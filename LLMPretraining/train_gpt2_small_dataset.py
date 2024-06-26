import torch
from torch.nn import functional as F
import time
import os

from llm_gpt2 import GPT, GPTConfig
from text_data_loader import DataLoaderLite

# --------------------------------------------------------------------------
# train GPT on a batch of data
torch.manual_seed(1337)
if torch.cuda.is_available:
    torch.cuda.manual_seed(1337)

device = "cuda" if torch.cuda.is_available else "cpu"
print("using device: ", device)


def train_gpt2(model, total_steps):
    total_batch_size = 524288  # 2^19, ~.5M tokens
    B, T = 8, 1024
    assert total_batch_size % (B * T) == 0, "total batch size is divisible by B * T"
    grad_accum_steps = total_batch_size // (B * T)
    print("total batch size: ", total_batch_size)
    print(f"calc gradient accumalate steps : {grad_accum_steps}")

    train_loader = DataLoaderLite(B=B, T=T)

    # trick 1 precision: set fp32 instead of float32
    # to reduce GPU memory, increase throughput (16800 -> 20400 token per sec)
    torch.set_float32_matmul_precision("high")

    model.to(device=device)
    # trick 3: compile : https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
    # throughput 38400 -> 39000, suitable for A100
    model = torch.compile(model)

    # algo trick 1: AdamW with (0.9, 0.95)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8
    )
    model.train()
    for step in range(total_steps):
        t0 = time.time()
        optimizer.zero_grad()

        # algo trick 3: batch size ~0.5M, accumlate the grad with 0.5M/(B*T) steps
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device=device), y.to(device=device)

            # trick 2: Auto Mixed Precison (https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
            # logits: bfloat16, weight, loss: float32, throughput: 20400 -> 38400
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = (
                loss / grad_accum_steps
            )  # accumlate with grad_accum_steps, need to avearge across all steps
            loss_accum += loss.detach()
            loss.backward()  # grad += grad(x,y)
        # algo trick 2: clip the gradient: https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html#torch.nn.utils.clip_grad_norm_
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # algo trick 4: cos learning rate schedule
        # lr = get_lr_schedule(step=step, max_steps=total_steps)
        # for params_group in optimizer.param_groups:
        #    params_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize()

        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps) / (
            t1 - t0
        )
        print(
            f"step: {step:4d} | loss: {loss_accum.item(): .6f} | norm: {norm: .4f} | dt: {dt: .2f} ms | {tokens_per_sec: .2f}"
        )

    # save checkpoint
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    check_point_file = os.path.join(log_dir, f"model_{total_steps}.pt")
    check_point = {"model": model.state_dict(), "config": model.config}
    torch.save(check_point, check_point_file)


# train GPT2 model from scratch
# trick 4: 50304 % 128 == 0, memory layout, throughput: 39000 -> 42400
model = GPT(GPTConfig(vocab_size=50304))
train_gpt2(model, 20)
