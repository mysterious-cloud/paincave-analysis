"""Neighborhood attention (NATTEN's 1D/2D qk-with-bias and av) in plain PyTorch, for Apple's GPU (MPS).

Everything stays 4-D, (batch*heads, positions, neighbours, features), so it runs as one gather and one batched
matmul: MPS has no fast path (and some reductions fail outright) for tensors of rank 6 and up.
"""
import functools
import torch


@functools.lru_cache(maxsize=256)
def _windows(length: int, kernel: int, dilation: int):
    """Per position: its K neighbours and K relative-bias indices, as NATTEN chooses them.

    Positions split into `dilation` interleaved groups; inside its group each position takes the K nearest
    members, the window shifted (not padded) at the group's edges.
    """
    i = torch.arange(length)
    g, j = i % dilation, i // dilation
    group_len = (length - g + dilation - 1) // dilation
    start = torch.minimum(torch.clamp(j - kernel // 2, min=0), torch.clamp(group_len - kernel, min=0))
    jj = start[:, None] + torch.arange(kernel)[None, :]
    return g[:, None] + jj * dilation, jj - j[:, None] + kernel - 1


@functools.lru_cache(maxsize=256)
def _tables1d(length, kernel, dilation, device):
    idx, pb = _windows(length, kernel, dilation)
    return idx.reshape(-1).to(device), pb.to(device)


@functools.lru_cache(maxsize=256)
def _tables2d(x, y, kernel, dilation, device):
    ix, px = _windows(x, kernel, dilation)
    iy, py = _windows(y, kernel, dilation)
    # (x, y, kx, ky) -> flat neighbour index into x*y, kx-major over the K*K window
    flat = ix[:, None, :, None] * y + iy[None, :, None, :]
    return flat.reshape(-1).to(device), px.to(device), py.to(device)


def _neighbours(t, flat, positions, window):
    """t: (N, positions, D) -> (N, positions, window, D)"""
    return t[:, flat].reshape(t.shape[0], positions, window, t.shape[-1])


def natten1dqkrpb(query, key, rpb, kernel_size, dilation):
    """query, key: (B, H, L, D); rpb: (H, 2K-1) or None -> (B, H, L, K)"""
    B, H, L, D = query.shape
    flat, pb = _tables1d(L, kernel_size, dilation, query.device)
    keys = _neighbours(key.reshape(B * H, L, D), flat, L, kernel_size)
    scores = (keys @ query.reshape(B * H, L, D, 1)).reshape(B, H, L, kernel_size)
    return scores if rpb is None else scores + rpb[:, pb]


def natten1dav(attn, value, kernel_size, dilation):
    """attn: (B, H, L, K); value: (B, H, L, D) -> (B, H, L, D)"""
    B, H, L, D = value.shape
    flat, _ = _tables1d(L, kernel_size, dilation, value.device)
    values = _neighbours(value.reshape(B * H, L, D), flat, L, kernel_size)
    return (attn.reshape(B * H, L, 1, kernel_size) @ values).reshape(B, H, L, D)


def natten2dqkrpb(query, key, rpb, kernel_size, dilation):
    """query, key: (B, H, X, Y, D); rpb: (H, 2K-1, 2K-1) or None -> (B, H, X, Y, K*K)"""
    B, H, X, Y, D = query.shape
    window = kernel_size * kernel_size
    flat, px, py = _tables2d(X, Y, kernel_size, dilation, query.device)
    keys = _neighbours(key.reshape(B * H, X * Y, D), flat, X * Y, window)
    scores = (keys @ query.reshape(B * H, X * Y, D, 1)).reshape(B, H, X, Y, window)
    if rpb is None:
        return scores
    bias = rpb[:, px[:, None, :, None], py[None, :, None, :]]  # (H, X, Y, K, K)
    return scores + bias.reshape(H, X, Y, window)


def natten2dav(attn, value, kernel_size, dilation):
    """attn: (B, H, X, Y, K*K); value: (B, H, X, Y, D) -> (B, H, X, Y, D)"""
    B, H, X, Y, D = value.shape
    window = kernel_size * kernel_size
    flat, _, _ = _tables2d(X, Y, kernel_size, dilation, value.device)
    values = _neighbours(value.reshape(B * H, X * Y, D), flat, X * Y, window)
    return (attn.reshape(B * H, X * Y, 1, window) @ values).reshape(B, H, X, Y, D)
