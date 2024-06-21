## Pretraining LLM 
Build GPT2/GPT3 from scratch, following from Andrej Karpathy's [video](https://www.youtube.com/watch?v=zduSFxRajkE).

We trained GPT-2 from scratch using 4 A10 GPUs with Distributed Data Parallel (DDP). 
```bash
torchrun --standalone --nproc_per_node=4 train_gpt2_ddp.py
```
The training utilized 1 billion tokens from the Fineweb dataset, processed in about 2000 batches, each containing approximately 0.5 million tokens. By employing computational and algorithmic optimization tricks, we achieved a 10X speedup, reducing the pre-training time to about an hour.
![Training Curve](./image/gpt2_convergence.png)
