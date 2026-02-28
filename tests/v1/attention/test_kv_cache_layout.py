# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.attention.backends.flash_attn_diffkv import (
    FlashAttentionDiffKVBackend,
)
from vllm.v1.attention.backends.flashinfer import (
    FlashInferBackend,
    _get_flashinfer_kv_layout,
)
from vllm.v1.attention.backends.utils import is_valid_kv_cache_layout


def test_flashattn_stride_order_for_nhd_with_layers(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.attention.backends.flash_attn.get_kv_cache_layout",
        lambda: "NHD",
    )

    assert FlashAttentionBackend.get_kv_cache_stride_order(
        include_num_layers_dimension=True
    ) == (2, 0, 1, 3, 4, 5)


def test_kv_nhd_stride_order_with_layers(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.attention.backends.flash_attn.get_kv_cache_layout",
        lambda: "KV_NHD",
    )
    monkeypatch.setattr(
        "vllm.v1.attention.backends.flashinfer.get_kv_cache_layout",
        lambda: "KV_NHD",
    )

    assert FlashAttentionBackend.get_kv_cache_stride_order(
        include_num_layers_dimension=True
    ) == (1, 2, 0, 3, 4, 5)
    assert FlashInferBackend.get_kv_cache_stride_order(
        include_num_layers_dimension=True
    ) == (1, 2, 0, 3, 4, 5)


def test_flashinfer_layout_mapping_for_kv_nhd(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.attention.backends.flashinfer.get_kv_cache_layout",
        lambda: "KV_NHD",
    )

    assert _get_flashinfer_kv_layout() == "NHD"


def test_kv_nhd_is_valid_layout():
    assert is_valid_kv_cache_layout("KV_NHD")


def test_flashattn_diffkv_kv_nhd_stride_order_with_layers(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.attention.backends.flash_attn_diffkv.get_kv_cache_layout",
        lambda: "KV_NHD",
    )

    assert FlashAttentionDiffKVBackend.get_kv_cache_stride_order(
        include_num_layers_dimension=True
    ) == (1, 0, 2, 3, 4)
