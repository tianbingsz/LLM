import torch
from llm_gpt2 import GPT, GPTConfig
import os

# load model weights from pretrained huggingface GPT2
torch.manual_seed(42)
torch.cuda.manual_seed(42)


def remove_prefix(state_dict, prefix):
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def load_model(check_point_file):
    checkpoint = torch.load(check_point_file)  # read checkpoint from log
    model = GPT(checkpoint["config"])  # GPT2 model with rand init weights
    state_dict = remove_prefix(checkpoint["model"], "_orig_mod.")
    print("load state dict to rand init GPT model")
    model.load_state_dict(state_dict)  # load weights from checkpoint
    return model


def generate_text_from_pretrained_hf_gpt2(
    load_pretrained_model=True, model_type="gpt2"
):
    device = "cuda" if torch.cuda.is_available else "cpu"
    # n_layer, n_head and n_embd are determined from model_type
    config_args = {
        "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
        "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
        "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
    }[model_type]
    config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
    config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
    config_args[
        "load_pretrained_model"
    ] = load_pretrained_model  # load model weights from HF pretrained model
    gpt_model = GPT(GPTConfig(**config_args))
    gpt_model.to(device=device)
    gpt_model.eval()
    gpt_model.generate(device=device, max_length=30)


# Load GPT2 model from pretrained hf GPT2 or rand init weight
# generate_text_from_pretrained_hf_gpt2(load_pretrained_model=False)
# generate_text_from_pretrained_hf_gpt2(load_pretrained_model=True)


def generate_text_from_gpt2(
    prompt="A Large Language Model is a step towards achieving Artificial General Intelligence,",
    check_point_file="log/model_1907.pt",
    device="cuda",
):
    print("load GPT model from checkpoint: ", check_point_file)
    gpt_model = load_model(check_point_file)
    gpt_model.to(device=device)
    gpt_model.eval()
    gpt_model.generate(prompt=prompt, device=device, max_length=30)


# load GPT2 model we trained
# log_dir = "log"
# check_point_file = os.path.join(log_dir, "model_1907.pt")
device = "cuda" if torch.cuda.is_available else "cpu"
prompt = "Hi, can you generate a random number,"
generate_text_from_gpt2(prompt=prompt, device=device)
