# -*- coding: utf-8 -*-
"""
DPO : https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2025_assignment5_supplement_safety_rlhf.pdf

1. Preference Data
"""

#!wget https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/helpful-base/train.jsonl.gz -O helpful-base.jsonl.gz

#!wget https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/harmless-base/train.jsonl.gz -O harmless-base.jsonl.gz

#!wget https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/helpful-online/train.jsonl.gz -O helpful-online.jsonl.gz

#!wget https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/helpful-rejection-sampled/train.jsonl.gz -O helpful-rejection-sampled.jsonl.gz

"""Write a function to load the Anthropic HH dataset. Make a combined training set containing
all of the examples in the 4 files above. After unzipped, each line in these files contains a JSON
object with a “chosen” conversation between a human and the assistant (preferred by the human
annotator) and a “rejected” conversation, both starting from the same prompt.
"""

import json
import re
from typing import List, Dict

def parse_turns(text: str) -> List[str]:
    """
    Split a conversation string into alternating turns by detecting 'Human:' and 'Assistant:'.
    Returns list of messages (role-tagged).
    """
    turns = re.split(r"\n*\s*(Human:|Assistant:)\s*", text.strip())
    merged = []
    for i in range(1, len(turns), 2):
        role = turns[i]
        content = turns[i+1].strip() if i+1 < len(turns) else ""
        merged.append((role, content))
    return merged


import gzip

def load_anthropic_hh_dataset(file_paths: Dict[str, str], max_per_file: int = 1000) -> List[Dict]:
    dataset = []

    for label, path in file_paths.items():
        count = 0
        print(f"Loading {path}...")
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                if count >= max_per_file:
                    break

                try:
                    example = json.loads(line)
                    chosen_turns = parse_turns(example['chosen'])
                    rejected_turns = parse_turns(example['rejected'])

                    dataset.append({
                        "instruction": chosen_turns[0][1],
                        "chosen": chosen_turns[1][1],
                        "rejected": rejected_turns[1][1],
                        "source": label
                    })
                    count += 1
                    if count % 100 == 0:
                        print(f"Loaded {count} examples so far. from {path}")

                except Exception as e:
                    print(f"[{label}] Skipping malformed line: {e}")
                    continue

    return dataset

file_paths = {
    "helpful-base": "helpful-base.jsonl.gz",
    "harmless-base": "harmless-base.jsonl.gz",
    "helpful-rejection-sampled": "helpful-rejection-sampled.jsonl.gz",
    "helpful-online": "helpful-online.jsonl.gz"
}

data = load_anthropic_hh_dataset(file_paths, max_per_file=250)
print(f"Loaded {len(data)} examples total.")


"""1.2 DataSet and DataLoader"""

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}{eos}
"""

import torch
from torch.utils.data import Dataset, DataLoader

class RLHFDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.end_of_token = tokenizer.eos_token_id

        self._format_and_tokenize_prompts()


    def _format_and_tokenize_prompts(self):
        self.encoded_prompts = []
        for line in self.data:
            instruction = line['instruction']
            chosen = line['chosen']
            rejected = line['rejected']

            prompt_chosen = ALPACA_PROMPT.format(instruction=instruction, input="", response=chosen, eos=self.end_of_token)
            prompt_rejected = ALPACA_PROMPT.format(instruction=instruction, input="", response=rejected, eos=self.end_of_token)

            encoded_prompt_chosen = self.tokenizer(
                prompt_chosen,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            encoded_prompt_rejected = self.tokenizer(
                prompt_rejected,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            self.encoded_prompts.append({
                'chosen_input_ids': encoded_prompt_chosen['input_ids'].squeeze(0),
                'chosen_attention_mask': encoded_prompt_chosen['attention_mask'].squeeze(0),
                'chosen_labels': encoded_prompt_chosen['input_ids'].clone().squeeze(0),
                'rejected_input_ids': encoded_prompt_rejected['input_ids'].squeeze(0),
                'rejected_attention_mask': encoded_prompt_rejected['attention_mask'].squeeze(0),
                'rejected_labels': encoded_prompt_rejected['input_ids'].clone().squeeze(0)
            })


    def __len__(self):
        return len(self.encoded_prompts)

    def __getitem__(self, idx):
        return self.encoded_prompts[idx]

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# DataLoader
dataset = RLHFDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

for batch in dataloader:
    print(batch.keys())
    print(batch['chosen_input_ids'].shape)
    print(batch['chosen_attention_mask'].shape)
    print(batch['chosen_labels'].shape)
    print(batch['rejected_input_ids'].shape)
    print(batch['rejected_attention_mask'].shape)
    print(batch['rejected_labels'].shape)
    break

"""2. DPO
You will now implement a training loop using DPO on the HH data. Unlike during SFT, we now have to
run two examples through the LMs (πref and πθ) to compute the loss, which takes significant GPU memory.
Thus, we will not try to batch our implementation, and use gradient accumulation, as done in SFT, to allow
for larger effective batch sizes. Similarly, we won’t be able to use AdamW unless we use other efficiency
tricks (such as quantization), so we will stick to the RMSprop optimizer (torch.optim.RMSprop), as also
done in the original DPO work. We suggest the following implementation path, which sacrifices maximal
performance for simplicity:

```
L = - \log \sigma(\beta \log(\pi_{\theta}(y_w|x)) - \log(\pi_{ref}(y_w|x))
- \beta{\log(\pi_{\theta}(y_l|x)) - \log(\pi_{ref}(y_l|x))}
)
```
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.nn.functional as F

def compute_log_probs(model, input_ids, labels, attention_mask=None):
    """
    Returns:
        - selected_log_probs: (B, T-1) — log-probs of each token (masked with -100s left as-is)
        - mask: (B, T-1) — 1.0 for valid positions, 0.0 otherwise
    """
    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits  # (B, T, V)
    shift_logits = logits[:, :-1, :].contiguous()     # (B, T-1, V)
    shift_labels = labels[:, 1:].contiguous()         # (B, T-1)

    # Mask: 1.0 where label is not -100
    mask = (shift_labels != -100).float()             # (B, T-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)   # (B, T-1, V)
    selected_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    return selected_log_probs, mask


def dpo_loss(model, ref_model, batch, beta=0.1):
    """
    Compute DPO loss using per-token log-probs
    L = - \log \sigma(\beta \log(\pi_{\theta}(y_w|x)) - \log(\pi_{ref}(y_w|x))
      - \beta{\log(\pi_{\theta}(y_l|x)) - \log(\pi_{ref}(y_l|x))})
    """
    # πθ (policy model) (B, T-1)
    logprob_chosen, mask_chosen = compute_log_probs(model, batch['chosen_input_ids'], batch['chosen_labels'], batch['chosen_attention_mask'])
    logprob_rejected, mask_rejected = compute_log_probs(model, batch['rejected_input_ids'], batch['rejected_labels'], batch['rejected_attention_mask'])

    # πref (reference model) (B, T-1)
    logprob_chosen_ref, _ = compute_log_probs(ref_model, batch['chosen_input_ids'], batch['chosen_labels'], batch['chosen_attention_mask'])
    logprob_rejected_ref, _ = compute_log_probs(ref_model, batch['rejected_input_ids'], batch['rejected_labels'], batch['rejected_attention_mask'])

    # Sequence-level log-probs: sum over valid tokens
    chosen_logprob_seq = (logprob_chosen * mask_chosen).sum(dim=1)    # (B,)
    rejected_logprob_seq = (logprob_rejected * mask_rejected).sum(dim=1)  # (B,)
    chosen_logprob_ref = (logprob_chosen_ref * mask_chosen).sum(dim=1)    # (B,)
    rejected_logprob_ref = (logprob_rejected_ref * mask_rejected).sum(dim=1)  # (B,)

    # DPO loss
    log_ratio = chosen_logprob_seq - rejected_logprob_seq
    ref_log_ratio = chosen_logprob_ref - rejected_logprob_ref
    dpo_argument = beta * (log_ratio - ref_log_ratio)

    loss = -F.logsigmoid(dpo_argument).mean()
    return loss

"""3. DPO Training"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def compute_reward_model_accuracy(model, dataloader, device=device, eps=1e-8):
    model.eval()
    total = 0
    correct = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        # If compute_log_probs returns tuple, grab first element
        log_prob_chosen, c_mask = compute_log_probs(model, batch['chosen_input_ids'], batch['chosen_labels'], batch['chosen_attention_mask'])

        log_prob_rejected, r_mask = compute_log_probs(model, batch['rejected_input_ids'], batch['rejected_labels'], batch['rejected_attention_mask'])

        # Sequence-level log-probs: sum over valid tokens
        chosen_prob_seq = (log_prob_chosen * c_mask).sum(dim=1) / (c_mask.sum(dim=1) + eps)  # (B,)
        rejected_prob_seq = (log_prob_rejected * r_mask).sum(dim=1) / (r_mask.sum(dim=1) + eps) # (B,)


        correct += (chosen_prob_seq > rejected_prob_seq).sum().item()
        total += log_prob_chosen.size(0)

    accuracy = correct / total if total > 0 else 0.0
    model.train()
    print(f"Validation reward model accuracy: {accuracy:.4f}")
    return accuracy

def dpo_training_loop(model, ref_model, dataloader, val_dataloader=None, lr=1e-5, max_epochs=1, device=device):
    model.to(device)
    ref_model.to(device)
    model.train()
    ref_model.eval()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    for epoch in range(max_epochs):
        for step, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            loss = dpo_loss(model, ref_model, batch)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if (step + 1) % 10 == 0:
              print(f"Epoch {epoch+1}/{max_epochs}, Step {step+1}/{len(dataloader)}, Loss: {loss.item():.4f}, Grad Norm: {norm:.4f}")

        # After each epoch, evaluate reward model accuracy if validation dataloader provided
        if val_dataloader is not None:
          compute_reward_model_accuracy(model, val_dataloader, device=device)

    # Optional: save model checkpoint here
    # model.save_pretrained("checkpoints/dpo")

def dpo_training_loop_gradient_accum(
    model,
    ref_model,
    dataloader,
    val_dataloader=None,
    grad_accum_steps=4,
    lr=1e-5,
    max_epochs=1,
    device='cuda',
    max_grad_norm=1.0,
):
    model.to(device)
    ref_model.to(device)
    model.train()
    ref_model.eval()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    optimizer.zero_grad()
    norm = 0.0  # initialize grad norm

    for epoch in range(max_epochs):
        loss_accum = 0.0
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # mix precision
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = dpo_loss(model, ref_model, batch)

            loss = loss / grad_accum_steps
            loss.backward()
            loss_accum += loss.item()

            if (step + 1) % grad_accum_steps == 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                print(f"Epoch {epoch+1}/{max_epochs}, Step {step+1}/{len(dataloader)}, "
                      f"Avg Loss: {loss_accum:.4f}, Grad Norm: {norm:.4f}")
                loss_accum = 0.0

        # After each epoch, evaluate reward model accuracy if validation dataloader provided
        if val_dataloader is not None:
          compute_reward_model_accuracy(model, val_dataloader, device=device)


    # Optional checkpoint saving
    # model.save_pretrained("checkpoints/dpo")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype = torch.bfloat16,
    device_map = 'auto'
)

ref_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype = torch.bfloat16,
    device_map = 'auto'
)

val_data = load_anthropic_hh_dataset(file_paths, max_per_file=50)
print(f"Loaded {len(val_data)} examples total.")

val_dataset = RLHFDataset(val_data, tokenizer)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)

dpo_training_loop(model, ref_model, dataloader, val_dataloader=val_dataloader, lr=1e-6, max_epochs=3, device=device)

dpo_training_loop_gradient_accum(model, ref_model, dataloader, val_dataloader=val_dataloader, lr=1e-6, max_epochs=2, device=device)