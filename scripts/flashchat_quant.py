#!/usr/bin/env python3
"""Quantization helpers for Flashchat model artifact builders."""

import numpy as np


def bf16_to_f32(data):
    arr = np.asarray(data)
    if arr.dtype == np.uint16:
        u16 = arr
    else:
        u16 = np.frombuffer(data, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def f32_to_bf16(arr):
    f32 = np.asarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32)
    return (u32 >> 16).astype(np.uint16)


def pack_4bit_rows(values):
    u4 = np.asarray(values, dtype=np.uint8)
    if u4.ndim != 2:
        raise ValueError("4-bit packing expects a 2D array")
    if u4.shape[1] % 8 != 0:
        raise ValueError("input dimension must be divisible by 8")
    if np.any(u4 > 15):
        raise ValueError("4-bit values must be in the range 0..15")

    packed = np.zeros((u4.shape[0], u4.shape[1] // 8), dtype=np.uint32)
    for i in range(8):
        packed |= u4[:, i::8].astype(np.uint32) << (4 * i)
    return packed


def unpack_4bit_rows(packed, in_dim):
    words = np.asarray(packed, dtype=np.uint32)
    if words.ndim != 2:
        raise ValueError("4-bit unpacking expects a 2D packed array")
    if in_dim % 8 != 0:
        raise ValueError("input dimension must be divisible by 8")
    if words.shape[1] != in_dim // 8:
        raise ValueError("packed shape does not match input dimension")

    u4 = np.zeros((words.shape[0], in_dim), dtype=np.uint8)
    for i in range(8):
        u4[:, i::8] = ((words >> (4 * i)) & 0xF).astype(np.uint8)
    return u4


def quantize_f32_to_4bit_affine_rows(values, group_size=64):
    f32 = np.asarray(values, dtype=np.float32)
    if f32.ndim != 2:
        raise ValueError("quantization expects a 2D array")
    if f32.shape[1] % group_size != 0:
        raise ValueError("input dimension must be divisible by group_size")
    if group_size % 8 != 0:
        raise ValueError("group_size must be divisible by 8")

    out_dim, in_dim = f32.shape
    num_groups = in_dim // group_size
    scales = np.zeros((out_dim, num_groups), dtype=np.float32)
    biases = np.zeros((out_dim, num_groups), dtype=np.float32)
    u4 = np.zeros((out_dim, in_dim), dtype=np.uint8)

    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        group = f32[:, start:end]
        min_vals = group.min(axis=1, keepdims=True)
        max_vals = group.max(axis=1, keepdims=True)
        scale = (max_vals - min_vals) / 15.0
        scale = np.where(scale < 1e-8, 1e-8, scale)
        quantized = np.rint((group - min_vals) / scale)
        u4[:, start:end] = np.clip(quantized, 0, 15).astype(np.uint8)
        scales[:, g:g + 1] = scale
        biases[:, g:g + 1] = min_vals

    return pack_4bit_rows(u4), f32_to_bf16(scales), f32_to_bf16(biases)


def dequantize_4bit_affine_rows(weight, scales_bf16, biases_bf16, in_dim, group_size=64):
    packed = np.asarray(weight, dtype=np.uint32)
    out_dim = packed.shape[0]
    num_groups = in_dim // group_size
    scales = bf16_to_f32(scales_bf16).reshape(out_dim, num_groups)
    biases = bf16_to_f32(biases_bf16).reshape(out_dim, num_groups)
    u4 = unpack_4bit_rows(packed, in_dim).astype(np.float32)
    out = np.empty((out_dim, in_dim), dtype=np.float32)

    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        out[:, start:end] = u4[:, start:end] * scales[:, g:g + 1] + biases[:, g:g + 1]
    return out


def convert_8bit_to_4bit(weight_u32, scales_bf16, biases_bf16, out_dim, in_dim, group_size=64):
    num_groups = in_dim // group_size
    scales_f32 = bf16_to_f32(scales_bf16).reshape(out_dim, num_groups)
    biases_f32 = bf16_to_f32(biases_bf16).reshape(out_dim, num_groups)

    u8_vals = np.zeros((out_dim, in_dim), dtype=np.uint8)
    for i in range(4):
        u8_vals[:, i::4] = (weight_u32[:, :in_dim // 4] >> (8 * i)) & 0xFF

    f32_vals = np.zeros((out_dim, in_dim), dtype=np.float32)
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        f32_vals[:, start:end] = (
            u8_vals[:, start:end].astype(np.float32) * scales_f32[:, g:g + 1]
            + biases_f32[:, g:g + 1]
        )

    return quantize_f32_to_4bit_affine_rows(f32_vals, group_size)


def split_qwen_gate_up_proj(gate_up):
    arr = np.asarray(gate_up)
    if arr.ndim < 2:
        raise ValueError("gate_up_proj tensor must have at least 2 dimensions")
    fused_dim = arr.shape[-2]
    if fused_dim % 2 != 0:
        raise ValueError("gate_up_proj fused dimension must be even")
    mid = fused_dim // 2
    return arr[..., :mid, :], arr[..., mid:, :]
