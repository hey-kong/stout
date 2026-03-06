# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.attention.backends.flash_attn_diffkv import FlashAttentionDiffKVBackend
from vllm.v1.attention.backends.flashinfer import FlashInferBackend
from vllm.v1.attention.backends.utils import get_kv_cache_layout, set_kv_cache_layout


def _set_layout(layout: str | None) -> None:
    set_kv_cache_layout(layout)
    get_kv_cache_layout.cache_clear()


def test_kv_nhd_stride_order_flash_attention_backends() -> None:
    _set_layout("KV_NHD")
    try:
        assert FlashAttentionBackend.get_kv_cache_stride_order() == (0, 1, 2, 3, 4)
        assert FlashAttentionBackend.get_kv_cache_stride_order(True) == (
            2,
            1,
            0,
            3,
            4,
            5,
        )

        assert FlashInferBackend.get_kv_cache_stride_order() == (0, 1, 2, 3, 4)
        assert FlashInferBackend.get_kv_cache_stride_order(True) == (1, 2, 0, 3, 4, 5)

        assert FlashAttentionDiffKVBackend.get_kv_cache_stride_order() == (0, 1, 2, 3)
        assert FlashAttentionDiffKVBackend.get_kv_cache_stride_order(True) == (
            1,
            0,
            2,
            3,
            4,
        )
    finally:
        _set_layout(None)
