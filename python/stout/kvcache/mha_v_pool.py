from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from stout.distributed import get_tp_info
from stout.utils import div_even

from .mha_pool import _create_buffer

if TYPE_CHECKING:
    from stout.hicache import HiCacheCounter


class MHAVCache:
    """
    Value-only cache pool for external cache.
    This class stores only V cache pages.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        layout: str,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._num_layers = num_layers
        self._device = device

        create_cache = lambda: _create_buffer(
            layout=layout,
            num_layers=num_layers,
            num_pages=num_pages,
            page_size=page_size,
            num_kv_heads=local_kv_heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )

        self.v_buffer = create_cache()
        self.storage_shape = (num_pages * page_size, local_kv_heads, head_dim)
        self.counter: HiCacheCounter | None = None

    def v_cache(self, index: int) -> torch.Tensor:
        return self.v_buffer[index]

    def set_hicache_counter(self, counter) -> None:
        self.counter = counter

    def get_kv_storage(self) -> Tuple[torch.Tensor | None, torch.Tensor]:
        return None, self.v_buffer

    def get_v_storage(self) -> torch.Tensor:
        return self.v_buffer

    def create_host_pool(self, num_pages: int, layout: str):
        num_layers, _, page_size, local_kv_heads, head_dim = self.v_buffer.shape
        return MHAVCache(
            num_kv_heads=local_kv_heads * get_tp_info().size,
            num_layers=num_layers,
            head_dim=head_dim,
            num_pages=num_pages,
            page_size=page_size,
            dtype=self.dtype,
            device=torch.device("cpu"),
            layout=layout,
        )

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        self.v_buffer[layer_id].view(self.storage_shape)[out_loc].copy_(v)
        if self.counter is not None:
            self.counter.wait(layer_id)

    def get_per_token_bytes(self) -> int:
        num_layers, _, _, local_kv_heads, head_dim = self.v_buffer.shape
        return num_layers * local_kv_heads * head_dim * self.v_buffer.element_size()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self.v_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
