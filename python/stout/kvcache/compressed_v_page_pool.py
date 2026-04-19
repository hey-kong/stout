from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import torch

from stout.distributed import get_tp_info
from stout.quant.quant_compressor import QuantMeta, QuantizedCompressor
from stout.quant import int4_ext
from stout.utils import div_even, init_logger

if TYPE_CHECKING:
    from stout.hicache import HiCacheCounter

logger = init_logger(__name__)


@dataclass
class CompressedVPageMeta:
    scale: torch.Tensor
    last_dim: int
    dtype: torch.dtype


class CompressedVPageCache:
    """Compressed value-only cache pool stored at page granularity (INT4 only)."""

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        packed_head_dim = (head_dim + 1) // 2

        self._num_layers = num_layers
        self._device = device
        self._dtype = dtype
        self._head_dim = head_dim
        self._page_size = page_size
        self._local_kv_heads = local_kv_heads

        # Store packed int4 payload in uint8.
        self.v_buffer = torch.empty(
            (num_layers, num_pages, page_size, local_kv_heads, packed_head_dim),
            device=device,
            dtype=torch.uint8,
        )
        # Per-page, per-layer scale with shape [num_pages, 1, num_layers, 1, 1].
        self.scale_buffer = torch.empty(
            (num_pages, 1, num_layers, 1, 1),
            device=device,
            dtype=torch.float32,
        )
        self.storage_shape = (num_pages * page_size, local_kv_heads, packed_head_dim)
        self.counter: HiCacheCounter | None = None
        self._compressor = QuantizedCompressor(bits=4)

    def set_hicache_counter(self, counter) -> None:
        self.counter = counter

    def get_kv_storage(self) -> Tuple[torch.Tensor | None, torch.Tensor]:
        return None, self.v_buffer

    def get_v_storage(self) -> torch.Tensor:
        return self.v_buffer

    def store_v_page(self, page_idx: int, v_page: torch.Tensor) -> CompressedVPageMeta:
        """
        Compress and store one V page.

        Args:
            page_idx: page index in this cache.
            v_page: tensor of shape [num_layers, page_size, local_kv_heads, head_dim].

        Returns:
            metadata required for decompression.
        """
        expected_shape = (self._num_layers, self._page_size, self._local_kv_heads, self._head_dim)
        if v_page.shape != expected_shape:
            raise ValueError(f"v_page must have shape {expected_shape}, got {tuple(v_page.shape)}")

        # [L, P, H, D] -> [1, P, L, H, D]
        page_input = v_page.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        packed, scale = int4_ext.quantize_pack_int4(page_input)

        self.v_buffer[:, page_idx].copy_(packed[0].permute(1, 0, 2, 3))
        page_scale = scale[0]
        self.scale_buffer[page_idx].copy_(page_scale)
        return CompressedVPageMeta(scale=self.scale_buffer[page_idx], last_dim=self._head_dim, dtype=v_page.dtype)

    def load_v_page(self, page_idx: int, meta: CompressedVPageMeta) -> torch.Tensor:
        """Load and decompress one V page as shape [num_layers, page_size, local_kv_heads, head_dim]."""
        packed_page = self.v_buffer[:, page_idx].permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        scale = meta.scale.to(packed_page.device, torch.float32).contiguous()
        page = int4_ext.dequant_unpack_int4(packed_page, scale, meta.last_dim).to(meta.dtype)[0]
        return page.permute(1, 0, 2, 3)

    def store_pages_from(
        self,
        src_v: torch.Tensor,
        src_pages: torch.Tensor,
        dst_pages: torch.Tensor,
    ) -> None:
        """
        Compress pages from an uncompressed V cache into this external compressed cache.

        Args:
            src_v: uncompressed V tensor with shape [num_layers, num_pages, page_size, local_kv_heads, head_dim].
            src_pages: source page indices in src_v.
            dst_pages: destination page indices in this cache.
        """
        if len(src_pages) != len(dst_pages):
            raise ValueError(f"len(src_pages)={len(src_pages)} must equal len(dst_pages)={len(dst_pages)}")

        for src_page, dst_page in zip(src_pages.tolist(), dst_pages.tolist()):
            self.store_v_page(int(dst_page), src_v[:, int(src_page)])

        num_tokens = len(src_pages) * self._page_size
        logger.info(
            f"External V Compress: {num_tokens:>5} tokens "
            f"({len(src_pages):>4} pages)"
        )

    def load_pages_to(
        self,
        dst_v: torch.Tensor,
        src_pages: torch.Tensor,
        dst_pages: torch.Tensor,
    ) -> None:
        """
        Decompress pages from this cache and write into uncompressed destination V cache.

        Args:
            dst_v: uncompressed V tensor with shape [num_layers, num_pages, page_size, local_kv_heads, head_dim].
            src_pages: source page indices in this compressed cache.
            dst_pages: destination page indices in dst_v.
        """
        if len(src_pages) != len(dst_pages):
            raise ValueError(f"len(src_pages)={len(src_pages)} must equal len(dst_pages)={len(dst_pages)}")

        # Normalize indices on cache device to enable efficient batched gather/decompress.
        src_pages_dev = src_pages.to(self.v_buffer.device, dtype=torch.long)
        dst_pages_dev = dst_pages.to(dst_v.device, dtype=torch.long)

        # Gather packed pages:
        # [L, num_pages, P, H, packed_D] -> [N, P, L, H, packed_D]
        packed_pages = self.v_buffer[:, src_pages_dev].permute(1, 2, 0, 3, 4).contiguous()
        metas = [
            QuantMeta(
                scale=self.scale_buffer[int(page)].to(self.v_buffer.device, torch.float32),
                last_dim=self._head_dim,
                dtype=dst_v.dtype,
            )
            for page in src_pages_dev.tolist()
        ]
        decompressed = self._compressor.allocate_batch_decompress_buffer(packed_pages, metas)
        self._compressor.batch_decompress(packed_pages, metas, out=decompressed)

        # [N, P, L, H, D] -> [L, N, P, H, D], then scatter into target pages.
        decompressed = decompressed.permute(2, 0, 1, 3, 4).contiguous()
        dst_v[:, dst_pages_dev].copy_(decompressed, non_blocking=True)

        num_tokens = len(src_pages) * self._page_size
        logger.info(
            f"External V Decompress: {num_tokens:>5} tokens "
            f"({len(src_pages):>4} pages)"
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
