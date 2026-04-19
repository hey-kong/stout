import torch
from typing import Optional
import logging
from dataclasses import dataclass

from . import int4_ext

logger = logging.getLogger(__name__)


@dataclass
class QuantMeta:
    scale: torch.Tensor
    last_dim: Optional[int] = None
    dtype: torch.dtype = torch.float16


class QuantizedCompressor:
    def __init__(self, bits: int = 4):
        assert bits in (4, 8)
        self.bits = bits

    @torch.inference_mode()
    def compress(self, v_cache: torch.Tensor):
        """
        v_cache layout: (num_pages, page_size, num_layers, num_kv_heads, head_dim)

        Quantization is done per-layer (num_layers dimension).
        """
        if v_cache.dim() < 4:
            raise ValueError(
                "v_cache must have shape (num_pages, page_size, num_layers, ..., head_dim)"
            )
        cv, meta = self._compress_tensor(v_cache, v_cache.dtype)
        return cv, meta

    def _compress_tensor(self, x: torch.Tensor, orig_dtype: torch.dtype):
        # int8
        if self.bits == 8:
            q, scale = self._quantize_int8(x)
            return q, QuantMeta(scale=scale, dtype=orig_dtype)

        # int4
        orig_D = x.shape[-1]
        packed, scale = int4_ext.quantize_pack_int4(x.contiguous())
        return packed, QuantMeta(scale=scale, last_dim=orig_D, dtype=orig_dtype)

    def _layerwise_abs_max(self, tensor: torch.Tensor):
        # tensor shape: (num_pages, page_size, num_layers, ...)
        if tensor.dim() < 3:
            raise ValueError(
                "tensor must have at least 3 dimensions (num_pages, page_size, num_layers, ...)"
            )
        layer_dim = 2
        reduce_dims = tuple(dim for dim in range(tensor.dim()) if dim != layer_dim)
        return tensor.abs().amax(dim=reduce_dims, keepdim=True)

    def _quantize_int8(self, tensor: torch.Tensor):
        t = tensor.float()

        # Symmetric per-layer quantization
        abs_max = self._layerwise_abs_max(t)
        scale = abs_max / 127.0
        scale = torch.clamp(scale, min=1e-6)

        q = torch.round(t / scale).clamp(-127, 127).to(torch.int8)
        return q, scale.to(torch.float32)

    @torch.inference_mode()
    def decompress(self, v_cache: torch.Tensor, meta: QuantMeta):
        """
        Decompress a V cache and restore layout:
        (num_pages, page_size, num_layers, num_kv_heads, head_dim)
        """
        if not isinstance(meta, QuantMeta):
            raise TypeError("meta must be QuantMeta")
        if v_cache.dim() < 4:
            raise ValueError(
                "v_cache must have shape (num_pages, page_size, num_layers, ..., head_dim/packed_head_dim)"
            )
        return self._decompress_tensor(v_cache, meta)

    def _decompress_tensor(self, x, meta: QuantMeta):
        scale = meta.scale.to(x.device, torch.float32)

        if self.bits == 8:
            out = x.float() * scale
            return out.to(meta.dtype)

        if meta.last_dim is None:
            raise ValueError("INT4 decompress requires last_dim in meta.")
        out = int4_ext.dequant_unpack_int4(x.contiguous(), scale.contiguous(), meta.last_dim)
        return out.to(meta.dtype)
