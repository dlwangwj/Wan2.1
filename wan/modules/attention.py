# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]

import xformers.ops as xops
from xformers.ops import memory_efficient_attention, fmha

def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.float16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
   
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes, f"dtype must be float16 or bfloat16, got {dtype}"
    assert q.device.type == "cuda" and q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # 预处理查询
    if q_lens is None:
        q = half(q.flatten(0, 1))  # [B*Lq, Nq, C1]
        q_lens = torch.full((b,), lq, dtype=torch.int32, device=q.device)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)], dim=0))

    # 预处理键和值
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.full((b,), lk, dtype=torch.int32, device=k.device)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)], dim=0))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)], dim=0))

    # 确保数据类型一致
    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)

    if q_scale is not None:
        q = q * q_scale

    # 调整键和值的头数以匹配查询
    n_q_heads = q.size(1)
    n_k_heads = k.size(1)
    if n_k_heads != n_q_heads:
        assert n_q_heads % n_k_heads == 0, "Nq must be divisible by Nk"
        repeat_factor = n_q_heads // n_k_heads
        k = k.repeat(1, repeat_factor, 1)
        v = v.repeat(1, repeat_factor, 1)

    # if window_size != (-1, -1):
    #     raise NotImplementedError("Sliding window attention not supported with xFormers")
    window_size = (-1, -1)

    # 生成块对角掩码
    q_lens_list = q_lens.cpu().tolist()
    k_lens_list = k_lens.cpu().tolist()

    if causal:
        
        attn_bias = fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(q_seqlen=q_lens_list)
    else:
        attn_bias = fmha.attn_bias.BlockDiagonalMask.from_seqlens(q_seqlen=q_lens_list, kv_seqlen=k_lens_list)

    # 添加虚拟批次维度以适应xFormers接口
    q = q.unsqueeze(0)  # [1, sum_q, nh, hd]
    k = k.unsqueeze(0)
    v = v.unsqueeze(0)

    # 调用xFormers的高效注意力实现
    x = xops.memory_efficient_attention(
        q, k, v,
        attn_bias=attn_bias,
        p=dropout_p,
        scale=softmax_scale,
        # deterministic=deterministic  # xFormers可能不支持此参数
    )

    # 移除虚拟批次维度并恢复原始形状
    x = x.squeeze(0).unflatten(0, (b, lq))  # [B, Lq, Nq, C2]

    return x.to(out_dtype)

# def flash_attention(
#     q,
#     k,
#     v,
#     q_lens=None,
#     k_lens=None,
#     dropout_p=0.,
#     softmax_scale=None,
#     q_scale=None,
#     causal=False,
#     window_size=(-1, -1),
#     deterministic=False,
#     dtype=torch.bfloat16,
#     version=None,
# ):
#     """
#     q:              [B, Lq, Nq, C1].
#     k:              [B, Lk, Nk, C1].
#     v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
#     q_lens:         [B].
#     k_lens:         [B].
#     dropout_p:      float. Dropout probability.
#     softmax_scale:  float. The scaling of QK^T before applying softmax.
#     causal:         bool. Whether to apply causal attention mask.
#     window_size:    (left right). If not (-1, -1), apply sliding window local attention.
#     deterministic:  bool. If True, slightly slower and uses more memory.
#     dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
#     """
#     half_dtypes = (torch.float16, torch.bfloat16)
#     assert dtype in half_dtypes
#     assert q.device.type == 'cuda' and q.size(-1) <= 256

#     # params
#     b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

#     def half(x):
#         return x if x.dtype in half_dtypes else x.to(dtype)

#     # preprocess query
#     if q_lens is None:
#         q = half(q.flatten(0, 1))
#         q_lens = torch.tensor(
#             [lq] * b, dtype=torch.int32).to(
#                 device=q.device, non_blocking=True)
#     else:
#         q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

#     # preprocess key, value
#     if k_lens is None:
#         k = half(k.flatten(0, 1))
#         v = half(v.flatten(0, 1))
#         k_lens = torch.tensor(
#             [lk] * b, dtype=torch.int32).to(
#                 device=k.device, non_blocking=True)
#     else:
#         k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
#         v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

#     q = q.to(v.dtype)
#     k = k.to(v.dtype)

#     if q_scale is not None:
#         q = q * q_scale

#     if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
#         warnings.warn(
#             'Flash attention 3 is not available, use flash attention 2 instead.'
#         )

#     # apply attention
#     if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
#         # Note: dropout_p, window_size are not supported in FA3 now.
#         x = flash_attn_interface.flash_attn_varlen_func(
#             q=q,
#             k=k,
#             v=v,
#             cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
#                 0, dtype=torch.int32).to(q.device, non_blocking=True),
#             cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
#                 0, dtype=torch.int32).to(q.device, non_blocking=True),
#             seqused_q=None,
#             seqused_k=None,
#             max_seqlen_q=lq,
#             max_seqlen_k=lk,
#             softmax_scale=softmax_scale,
#             causal=causal,
#             deterministic=deterministic)[0].unflatten(0, (b, lq))
#     else:
#         assert FLASH_ATTN_2_AVAILABLE
#         x = flash_attn.flash_attn_varlen_func(
#             q=q,
#             k=k,
#             v=v,
#             cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
#                 0, dtype=torch.int32).to(q.device, non_blocking=True),
#             cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
#                 0, dtype=torch.int32).to(q.device, non_blocking=True),
#             max_seqlen_q=lq,
#             max_seqlen_k=lk,
#             dropout_p=dropout_p,
#             softmax_scale=softmax_scale,
#             causal=causal,
#             window_size=window_size,
#             deterministic=deterministic).unflatten(0, (b, lq))

#     # output
#     return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
