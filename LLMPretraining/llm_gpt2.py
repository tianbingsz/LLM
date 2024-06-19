import string
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

from transformers import GPT2LMHeadModel
from dataclasses import dataclass


# --------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """
    Causal Self Attention
    attn.c_attn.weight torch.Size([768, 2304]), attn.c_attn.bias torch.Size([2304])
    attn.c_proj.weight torch.Size([768, 768]), attn.c_proj.bias torch.Size([768])
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        """
        self attention
        calculate query, key, values for all heads in batch and move head forward to be the batch dim
        nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        input x: (B, T, C)
        return: (B, T, C), C = 768
        """
        (
            B,
            T,
            C,
        ) = x.shape  # batch size, sequence length, embedding dimensionality (n_embd)
        qkv = self.c_attn(x)  # (B, T, 3 * 768)
        q, k, v = qkv.split(self.n_embd, dim=2)  # q, k, v : (B, T, 768)
        q = q.view(B, T, self.n_head, C // self.n_head).permute(
            0, 2, 1, 3
        )  # (B, T, 768) => (B, n_head = 12, T, h_dim = 64)
        k = k.view(B, T, self.n_head, C // self.n_head).permute(
            0, 2, 1, 3
        )  # (B, T, 768) => (B, n_head = 12, T, h_dim = 64)
        v = v.view(B, T, self.n_head, C // self.n_head).permute(
            0, 2, 1, 3
        )  # (B, T, 768) => (B, n_head = 12, T, h_dim = 64)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )  # (B, n_head, T, h_dim), trick: flashattention
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, C)  # (B, T, C)
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """
    MLP layer
    mlp.c_fc.weight torch.Size([768, 3072]), mlp.c_fc.bias torch.Size([3072])
    mlp.c_proj.weight torch.Size([3072, 768]), mlp.c_proj.bias torch.Size([768])
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        # (B, T, n_embd) => (B, T, n_embd)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """
    A layer of masked self-attention and MLP with layer norm in between
    Layer Norm
    ln_1.weight torch.Size([768]), ln_1.bias torch.Size([768])
    CausalSelfAttention:
    attn.c_attn.weight torch.Size([768, 2304]), attn.c_attn.bias torch.Size([2304])
    attn.c_proj.weight torch.Size([768, 768]), attn.c_proj.bias torch.Size([768])
    Layer Norm
    ln_2.weight torch.Size([768]), ln_2.bias torch.Size([768])
    MLP:
    mlp.c_fc.weight torch.Size([768, 3072]), mlp.c_fc.bias torch.Size([3072])
    mlp.c_proj.weight torch.Size([3072, 768]), mlp.c_proj.bias torch.Size([768])
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        """
        input x: (B, T, n_embd)
        output: (B, T, V)
        """
        # residual nets
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    model_type: string = "gpt2"  # model type GPT2
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension
    load_pretrained_model: bool = (
        False  # load weight from pretrained model or init from random
    )


class GPT(nn.Module):
    """
    GPT2 model
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # (B, T, V) => (B, T, n_embd), V = vocab_size, T = 1024, n_embd = 768
        self.transformer = nn.ModuleDict(
            dict(
                # words token embedding
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # (50527, 768)
                # words pos embedding
                wpe=nn.Embedding(config.block_size, config.n_embd),  # (1024, 768)
                # Nx layers of masked self-attention and MLP
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                # layer norm
                ln_f=nn.LayerNorm(config.n_embd),  # (768)
            )
        )
        # linear transformation to logits: (B, T, n_embd) => (B, T, V)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing, input embedding = ouput linear transform, reduce parameters
        self.transformer.wte.weight = self.lm_head.weight

        if config.load_pretrained_model is True:  # load weights from pretrained GPT2
            print("load model weights from pretrained huggingface GPT2 ...")
            self.copy_weights_from_hf_GPT2(config.model_type)
        else:  # random init weights
            print("train GPT2 from scratch, random init the model parameters ...")
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        input token idx: (B, T), target : None or (B, T)
        outupt logits: (B, T, vocab_size)
        """
        # idx is of shape (B, T)
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(
        self,
        device="cuda",
        prompt="Hello, I'm a language model, ",
        num_sequence=5,
        max_length=20,
    ):
        enc = tiktoken.get_encoding("gpt2")  # use gpt2 tokenizer
        tokens = enc.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
        tokens = tokens.unsqueeze(0).repeat(num_sequence, 1)  # (5, 8), batch_size = 5
        x = tokens.to(device)  # (B, T)

        # generate max len of tokens from prompt
        for i in range(max_length):
            # forward the model to get logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits = self(x)[0]  # (B, T, vocab_size)
                    # take the logits at the last pos
                    logits = logits[:, -1, :]  # (B, vocab_size)
                    # calc the prob across the vocab
                    probs = F.softmax(logits, dim=-1)
                    # pick the top tokens
                    top_probs, top_idx = torch.topk(probs, 50, dim=-1)
                    # greedy sampling to get top 1 idx for each batch
                    idx = torch.multinomial(top_probs, 1)  # (B, 1)
                    # gather the corresponding indices
                    xcol = torch.gather(top_idx, -1, idx)  # (B, 1)
                    x = torch.cat((x, xcol), dim=1)  # (B, T+1)

        # decode the generated tokens to texts
        B, T = x.shape
        print(f"generate text for {B} sequences, each with max len {T}")
        for i in range(B):
            tokens = x[i, :].tolist()
            print(">", enc.decode(tokens))

    def copy_weights_from_hf_GPT2(self, model_type):
        sd = self.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


# ----------------------------------------------------------------------------------------------
