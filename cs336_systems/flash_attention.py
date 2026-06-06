from __future__ import annotations

import math

import triton
import triton.language as tl
import torch

from cs336_systems.flash_attention_backward import flash_attention_backward_pytorch


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # batch 和 query 维度的 确定唯一一个 program instance，在二维网络中拿到自己的坐标
    query_tile_index = tl.program_id(1) 
    batch_index = tl.program_id(0)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0), # 这是给 Triton 的内存访问顺序提示
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # boundary_check=(0, 1) 表示两个维度都做越界检查。越界的位置用 0 填充。
    q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE) # 计算当前 tile 中每个 query 的全局索引，用于后续的 causal mask 计算

    m_i = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for key_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_offsets = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where(k_offsets[None, :] < N_KEYS, scores, -float("inf"))
        if is_causal:
            scores = tl.where(q_offsets[:, None] >= k_offsets[None, :], scores, -1.0e6)

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = alpha[:, None] * acc
        acc = tl.dot(p.to(V_block_ptr.type.element_ty), v, acc=acc)
        m_i = m_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    output = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)

    tl.store(O_block_ptr, output.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, lse, boundary_check=(0,))



class FlashAttentionTritonAutogradFunction(torch.autograd.Function):
    Q_TILE_SIZE = 16
    K_TILE_SIZE = 16

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        if not Q.is_cuda or not K.is_cuda or not V.is_cuda:
            raise ValueError("FlashAttentionTritonAutogradFunction requires CUDA tensors")

        batch_shape = Q.shape[:-2] 
        n_queries = Q.shape[-2]
        n_keys = K.shape[-2]
        d = Q.shape[-1]

        if K.shape[:-2] != batch_shape or V.shape[:-2] != batch_shape:
            raise ValueError("Q, K, and V must have matching batch dimensions")
        if K.shape[-1] != d:
            raise ValueError("Q and K must have the same embedding dimension")
        if V.shape[-2] != n_keys:
            raise ValueError("K and V must have the same sequence length")
        if V.shape[-1] != d:
            raise ValueError("This Triton forward kernel expects V.shape[-1] == Q.shape[-1]")

        batch_size = math.prod(batch_shape) if batch_shape else 1
        q = Q.reshape(batch_size, n_queries, d)
        k = K.reshape(batch_size, n_keys, d)
        v = V.reshape(batch_size, n_keys, d)

        score_dtype = torch.promote_types(Q.dtype, K.dtype)
        out_dtype = torch.promote_types(score_dtype, V.dtype)
        output = torch.empty((batch_size, n_queries, d), dtype=out_dtype, device=Q.device)
        L = torch.empty((batch_size, n_queries), dtype=torch.float32, device=Q.device)

        grid = (batch_size, triton.cdiv(n_queries, FlashAttentionTritonAutogradFunction.Q_TILE_SIZE))
        flash_fwd_kernel[grid](
            q,
            k,
            v,
            output,
            L,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            L.stride(0),
            L.stride(1),
            n_queries,
            n_keys,
            1.0 / math.sqrt(d),
            D=d,
            Q_TILE_SIZE=FlashAttentionTritonAutogradFunction.Q_TILE_SIZE,
            K_TILE_SIZE=FlashAttentionTritonAutogradFunction.K_TILE_SIZE,
            is_causal=bool(is_causal),
        )

        output = output.reshape(*batch_shape, n_queries, d)
        L = L.reshape(*batch_shape, n_queries).to(score_dtype)
        ctx.save_for_backward(L, Q, K, V, output)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, output = ctx.saved_tensors
        dQ, dK, dV = flash_attention_backward_pytorch(Q, K, V, output, dO, L, ctx.is_causal)
        return dQ, dK, dV, None



class FlashAttentionPytorchAutogradFunction(torch.autograd.Function):
    BLOCK_M = 16
    BLOCK_N = 16

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch_shape = Q.shape[:-2]
        n_queries = Q.shape[-2]
        n_keys = K.shape[-2]
        d = Q.shape[-1]
        d_v = V.shape[-1]

        if K.shape[:-2] != batch_shape or V.shape[:-2] != batch_shape:
            raise ValueError("Q, K, and V must have matching batch dimensions")
        if K.shape[-1] != d:
            raise ValueError("Q and K must have the same embedding dimension")
        if V.shape[-2] != n_keys:
            raise ValueError("K and V must have the same sequence length")

        batch_size = math.prod(batch_shape) if batch_shape else 1
        q = Q.reshape(batch_size, n_queries, d)
        k = K.reshape(batch_size, n_keys, d)
        v = V.reshape(batch_size, n_keys, d_v)

        score_dtype = torch.promote_types(Q.dtype, K.dtype)
        acc_dtype = torch.float32 if score_dtype in (torch.float16, torch.bfloat16) else score_dtype
        out_dtype = torch.promote_types(score_dtype, V.dtype)
        scale = 1.0 / math.sqrt(d)

        o = torch.empty((batch_size, n_queries, d_v), dtype=acc_dtype, device=Q.device)
        lse = torch.empty((batch_size, n_queries), dtype=acc_dtype, device=Q.device)

        for q_start in range(0, n_queries, FlashAttentionPytorchAutogradFunction.BLOCK_M):
            q_end = min(q_start + FlashAttentionPytorchAutogradFunction.BLOCK_M, n_queries)
            q_block = q[:, q_start:q_end].to(acc_dtype)
            q_len = q_end - q_start

            m_i = torch.full((batch_size, q_len), -torch.inf, dtype=acc_dtype, device=Q.device)
            l_i = torch.zeros((batch_size, q_len), dtype=acc_dtype, device=Q.device)
            o_i = torch.zeros((batch_size, q_len, d_v), dtype=acc_dtype, device=Q.device)

            for k_start in range(0, n_keys, FlashAttentionPytorchAutogradFunction.BLOCK_N):
                k_end = min(k_start + FlashAttentionPytorchAutogradFunction.BLOCK_N, n_keys)
                k_block = k[:, k_start:k_end].to(acc_dtype)
                v_block = v[:, k_start:k_end].to(acc_dtype)

                scores = torch.matmul(q_block, k_block.transpose(-1, -2)) * scale
                if is_causal:
                    q_idx = torch.arange(q_start, q_end, device=Q.device)[:, None]
                    k_idx = torch.arange(k_start, k_end, device=Q.device)[None, :]
                    causal_mask = q_idx >= k_idx
                    scores = torch.where(causal_mask[None, :, :], scores, torch.full_like(scores, -1e6))

                m_ij = torch.max(scores, dim=-1).values
                m_new = torch.maximum(m_i, m_ij)
                alpha = torch.exp(m_i - m_new)
                p = torch.exp(scores - m_new[:, :, None]) # 加一维，便于与score做差（广播）

                l_i = alpha * l_i + torch.sum(p, dim=-1)
                o_i = alpha[:, :, None] * o_i + torch.matmul(p, v_block)
                m_i = m_new

            o[:, q_start:q_end] = o_i / l_i[:, :, None]
            lse[:, q_start:q_end] = m_i + torch.log(l_i)

        output = o.reshape(*batch_shape, n_queries, d_v).to(out_dtype)
        L = lse.reshape(*batch_shape, n_queries).to(score_dtype)
        ctx.save_for_backward(L, Q, K, V, output)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, output = ctx.saved_tensors
        dQ, dK, dV = flash_attention_backward_pytorch(Q, K, V, output, dO, L, ctx.is_causal)
        return dQ, dK, dV, None
