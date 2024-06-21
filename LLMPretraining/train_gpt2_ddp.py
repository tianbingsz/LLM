import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import time
import os

from llm_gpt2 import GPT, GPTConfig
from text_data_loader import FineWebDistributedDataLoaderLite
from optimizer import configure_optimizer, get_lr_schedule
from utils import eval_gpt2

# --------------------------------------------------------------------------
# train GPT on a batch of data with DDP on 4 GPUs
# torchrun --standalone --nproc_per_node=4 train_gpt2_ddp.py
# with 4 GPUs, throughput increase from 46800 -> 173,000 tokens per sec

assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
# initializes the default process group for DDP
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
print("run ddp ?", ddp)
init_process_group(backend="nccl")
# the global rank of a process in the DDP setup. It is a unique identifier for each process across all nodes.
ddp_rank = int(os.environ["RANK"])  # in [0,3]
# local rank of a process on a specific node. It identifies the GPU to be used by the process on the current node.
ddp_local_rank = int(os.environ["LOCAL_RANK"])  # in [0,3]
# the total number of processes across all nodes. Each process typically corresponds to one GPU.
ddp_world_size = int(os.environ["WORLD_SIZE"])  # 4 A10

# master_process == (ddp_rank == 0)  # this process will do logging, checkpointing etc

device = f"cuda:{ddp_local_rank}"
torch.cuda.set_device(device)
device_type = "cuda" if device.startswith("cuda") else "cpu"  # should be cuda here
print("device_type, using device: ", device_type, device)

torch.manual_seed(1337)
if torch.cuda.is_available:
    torch.cuda.manual_seed(1337)

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:  # open for writing to clear the file
    pass
log_eval_file = os.path.join(log_dir, f"log_eval.txt")
with open(log_eval_file, "w") as f:
    pass


def train_gpt2(
    model,
    data_root="edu_fineweb1B",
    warmup_steps=70,
    total_steps=1907,
    eval_per_steps=250,
):
    total_batch_size = 524288  # 2^19, ~.5M tokens
    B, T = 8, 1024
    assert (
        total_batch_size % (B * T * ddp_world_size) == 0
    ), "total batch size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if ddp_rank == 0:  # master_process
        print("total batch size: ", total_batch_size)
        print(f"calc gradient accumalate steps : {grad_accum_steps}")

    train_loader = FineWebDistributedDataLoaderLite(
        B=B,
        T=T,
        process_rank=ddp_rank,
        num_process=ddp_world_size,
        split="train",
        data_root=data_root,
    )

    eval_loader = FineWebDistributedDataLoaderLite(
        B=B,
        T=T,
        process_rank=ddp_rank,
        num_process=ddp_world_size,
        split="val",
        data_root=data_root,
    )

    # trick 1 precision: set fp32 instead of float32
    # to reduce GPU memory, increase throughput (16800 -> 20400 token per sec)
    torch.set_float32_matmul_precision("high")

    model.to(device=device)
    # trick 3: compile : https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
    # throughput 38400 -> 39000, suitable for A100
    model = torch.compile(model)
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module  # always contains the "raw" unwrapped model

    # algo trick 1: AdamW with (0.9, 0.95)
    optimizer = configure_optimizer(
        raw_model, weight_decay=0.1, learning_rate=6e-4, device_type=device_type
    )
    for step in range(total_steps):
        t0 = time.time()
        last_step = step == total_steps - 1
        # eval model per (e.g. 250) steps print at master process
        if (step > 0 and step % eval_per_steps == 0) or last_step:
            model.eval()
            eval_loader.reset()
            eval_loss = eval_gpt2(
                model,
                eval_loader=eval_loader,
                device=device,
                device_type=device_type,
                ddp=True,
            )
            if ddp_rank == 0:
                print(f"step: {step: 4d}, validation loss: {eval_loss: .6f}")
                with open(log_eval_file, "a") as f:
                    f.write(f"step: {step:4d} | loss: {eval_loss: .6f}\n")

        # train model per step
        model.train()
        optimizer.zero_grad()

        # algo trick 3: batch size ~0.5M, accumlate the grad with 0.5M/(B*T) steps
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device=device), y.to(device=device)

            # trick 2: Auto Mixed Precison (https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
            # logits: bfloat16, weight, loss: float32, throughput: 20400 -> 38400
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = (
                loss / grad_accum_steps
            )  # accumlate with grad_accum_steps, need to avearge across all steps
            loss_accum += loss.detach()
            # trick 5: only all_reduce the gradients on all GPUs at the last micro_step, to reduce unnecessary communications
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            loss.backward()  # grad += grad(x,y)
        # AVG the loss across all GPUs
        dist.all_reduce(loss_accum, dist.ReduceOp.AVG)
        # algo trick 2: clip the gradient: https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html#torch.nn.utils.clip_grad_norm_
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # algo trick 4: cos learning rate schedule
        lr = get_lr_schedule(
            step=step, warmup_steps=warmup_steps, max_steps=total_steps
        )
        for params_group in optimizer.param_groups:
            params_group["lr"] = lr
        optimizer.step()
        torch.cuda.synchronize()

        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = (
            train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        ) / (t1 - t0)
        if ddp_rank == 0:  # master process
            print(
                f"step: {step:4d} | loss: {loss_accum.item(): .6f} | lr: {lr: .4e} | norm: {norm: .4f} | dt: {dt: .2f} ms | {tokens_per_sec: .2f}"
            )
            with open(log_file, "a") as f:
                f.write(f"step: {step:4d} | loss: {loss_accum.item(): .6f}\n")

    # save checkpoint
    if ddp_rank == 0:  # master process
        check_point_file = os.path.join(log_dir, f"model_{total_steps}.pt")
        check_point = {"model": raw_model.state_dict(), "config": raw_model.config}
        torch.save(check_point, check_point_file)

    # clean up
    destroy_process_group()


# torchrun --standalone --nproc_per_node=4 train_gpt2_ddp.py
# train GPT2 model from scratch
# trick 4: 50304 % 128 == 0, memory layout, throughput: 39000 -> 42400
model = GPT(GPTConfig(vocab_size=50304))
data_root = "edu_fineweb10B"  # 10B tokens
warmup_steps = 715
max_steps = 19073  # 10B tokens ~1 epoch, each step process a batch of 0.5M tokens
train_gpt2(
    model,
    data_root=data_root,
    warmup_steps=warmup_steps,
    total_steps=max_steps,
    eval_per_steps=250,
)
