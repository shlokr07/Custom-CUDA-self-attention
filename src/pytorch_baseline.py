import torch
import numpy as np
import time

# === Load Embedding Tensor ===
E = np.fromfile(r"data/embeddings1k.bin", dtype=np.float32).reshape(-1, 32, 300)  # [B, T, D]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
E = torch.tensor(E, dtype=torch.float32).to(device)  # [B, T, D]

B, T, D = E.shape

# === Set Number of Heads ===
num_heads = 1

if D % num_heads != 0:
    raise ValueError(f"Embedding dim {D} must be divisible by num_heads={num_heads}")

# === MultiheadAttention requires input shape [T, B, D] ===
E = E.transpose(0, 1)  # [T, B, D]

# === Define Layer ===
mha = torch.nn.MultiheadAttention(embed_dim=D, num_heads=num_heads, batch_first=False).to(device)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
Z, attn_weights = mha(E, E, E)
end.record()

torch.cuda.synchronize()
print(f"Time taken: {start.elapsed_time(end):.3f} ms")