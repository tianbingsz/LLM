import torch
import os

from text_data_loader import DataLoaderLite, FineWebDistributedDataLoaderLite
from utils import load_model, eval_gpt2

device_type = "cuda" if torch.cuda.is_available else "cpu"
log_dir = "log"
check_point_file = os.path.join(log_dir, "model_1907.pt")
model = load_model(check_point_file)
model.to(device="cuda")
# eval GPT2 we trained (with DDP=False)
# eval_loader = DataLoaderLite(B=8, T=1024)
eval_loader = FineWebDistributedDataLoaderLite(B=8, T=1024, split="val")
eval_loss = eval_gpt2(model, eval_loader=eval_loader, ddp=False)
print("eval loss: ", eval_loss)
