import torch
import tiktoken


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("input.txt") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"load {len(self.tokens)} tokens")
        assert len(self.tokens > B * T), "num of tokens at least has one batch"
        print(f"one epoch = {len(self.tokens) // (B * T) } batches")
        # buf cur pos
        self.cur_pos = 0

    def next_batch(self):
        B, T = self.B, self.T
        # try advancing to next batch, reset if out of bound
        if self.cur_pos + B * T + 1 > len(self.tokens):
            self.cur_pos = 0
        buf = self.tokens[self.cur_pos : self.cur_pos + B * T + 1]
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)  # targets
        # advance the pos
        self.cur_pos += B * T
        return x, y
