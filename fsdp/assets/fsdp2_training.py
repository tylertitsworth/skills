"""Minimal FSDP2 training example with mixed precision and checkpointing.

Shows the recommended FSDP2 pattern:
  1. Build model on meta device
  2. Apply fully_shard bottom-up
  3. Materialize parameters
  4. Train with torch.compile
  5. Save distributed checkpoint

Usage:
    torchrun --nproc_per_node=4 fsdp2_training.py

Requires: PyTorch 2.4+
"""

import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 1024, nhead: int = 16, dim_ff: int = 4096):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(
        self, vocab_size: int = 32000, d_model: int = 1024, n_layers: int = 12
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Device mesh for FSDP
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    # Mixed precision policy
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )

    # Build model
    model = SimpleTransformer()
    model = model.to(local_rank)

    # Apply FSDP2 bottom-up: layers first, then root
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

    # Compile after FSDP wrapping
    model = torch.compile(model)

    # Optimizer on DTensor parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Training loop
    for step in range(100):
        x = torch.randint(0, 32000, (4, 512), device=f"cuda:{local_rank}")
        logits = model(x)
        loss = logits.sum()  # dummy loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if rank == 0 and step % 10 == 0:
            print(f"Step {step}: loss={loss.item():.4f}")

    # Save distributed checkpoint
    if rank == 0:
        print("Saving checkpoint...")
    state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    dcp.save(state_dict, checkpoint_id="/tmp/fsdp2_ckpt")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
