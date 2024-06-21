import torch
import tiktoken

import os
import numpy as np


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
        assert len(self.tokens) > B * T, "num of tokens at least has one batch"
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


class DistributedDataLoaderLite:
    """
    DDP: distributed replica into different num_process GPUs
    current GPU process: process_rank
    """

    def __init__(self, B, T, process_rank=0, num_process=1):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_process = num_process

        with open("input.txt") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        assert len(self.tokens) > B * T, "num of tokens at least has one batch"
        if process_rank == 0:  # master process
            print(f"load {len(self.tokens)} tokens")
            print(f"one epoch = {len(self.tokens) // (B * T * num_process) } batches")
        # buf cur pos for rank
        self.cur_pos = B * T * process_rank

    def next_batch(self):
        B, T = self.B, self.T
        # try advancing to next batch, reset if out of bound
        if self.cur_pos + B * T * self.num_process + 1 > len(self.tokens):
            self.cur_pos = B * T * self.process_rank
        buf = self.tokens[self.cur_pos : self.cur_pos + B * T + 1]
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)  # targets
        # advance the pos for current process, each advance B*T
        self.cur_pos += B * T * self.num_process
        return x, y


class FineWebDistributedDataLoaderLite:
    """
    FineWeb dataset, we use 100M tokens
    DDP: distributed replica into different num_process GPUs
    current GPU process: process_rank
    """

    def __init__(
        self,
        B,
        T,
        process_rank=0,
        num_process=1,
        split="train",
        data_root="edu_fineweb1B",
    ):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_process = num_process
        assert split in {"train", "val"}

        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]  # train or val
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards

        assert len(shards) > 0, f"no shards found in {split}"
        if process_rank == 0:  # master process
            print(f"using dataset: {data_root}, found {len(shards)} for {split}")

        self.reset()

    def reset(self):
        self.cur_shard = 0
        tokens, status = self._load_tokens(self.shards[self.cur_shard])
        # skip tokens if status is False
        if status is True:
            self.tokens = tokens
        self.cur_pos = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        # try advancing to next batch, reset and load tokens from next shard if out of bound
        if self.cur_pos + B * T * self.num_process + 1 > len(self.tokens):
            self.cur_shard = (self.cur_shard + 1) % len(self.shards)
            tokens, status = self._load_tokens(self.shards[self.cur_shard])
            # skip tokens if status is False
            if status is True:
                self.tokens = tokens
            self.cur_pos = B * T * self.process_rank

        buf = self.tokens[self.cur_pos : self.cur_pos + B * T + 1]
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)  # targets
        # advance the pos for current process, each advance B*T
        self.cur_pos += B * T * self.num_process
        return x, y

    def _load_tokens(self, filename):
        """
        load tokens from npy file and convert to tensors
        """
        npt = np.load(filename)
        npt = npt.astype(np.int32)
        ptt = torch.tensor(npt, dtype=torch.long)
        status = len(ptt) > self.B * self.T
        # assert len(ptt) > self.B * self.T, f"num of tokens {len(ptt)} at least has one batch {self.B * self.T}"
        # print(f"load {len(ptt)} tokens")
        # print(f"one epoch = {len(ptt) // (self.B * self.T * self.num_process) } batches")

        return ptt, status
