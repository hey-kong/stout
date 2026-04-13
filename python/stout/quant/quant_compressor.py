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

    @torch.inference_mode()
    def batch_decompress(self, v_caches: torch.Tensor, metas, out: Optional[torch.Tensor] = None):
        """
        Decompress batched V caches and restore layout:
        (num_blocks, num_pages, page_size, num_layers, num_kv_heads, head_dim)

        Args:
            v_caches: Tensor shaped
                (num_blocks, num_pages, page_size, num_layers, ..., head_dim/packed_head_dim)
            metas: sequence of QuantMeta with length == num_blocks,
                   or a single QuantMeta to be reused for all blocks.
            out: Optional preallocated output tensor returned by
                 `allocate_batch_decompress_buffer`. If provided,
                 batch_decompress only performs decompression and write-back.
                 If None, keeps original logic by decompressing each block then stacking.
        """
        if not isinstance(v_caches, torch.Tensor) or v_caches.dim() < 5:
            raise ValueError(
                "v_caches must have shape (num_blocks, num_pages, page_size, num_layers, ..., head_dim/packed_head_dim)"
            )

        num_blocks = v_caches.size(0)
        if isinstance(metas, QuantMeta):
            metas = [metas] * num_blocks
        if len(metas) != num_blocks:
            raise ValueError(f"len(metas)={len(metas)} must equal num_blocks={num_blocks}")
        if any(not isinstance(m, QuantMeta) for m in metas):
            raise TypeError("each meta in metas must be QuantMeta")

        if out is None:
            return torch.stack([
                self.decompress(v_caches[i], metas[i])
                for i in range(num_blocks)
            ], dim=0)

        for i in range(num_blocks):
            out[i].copy_(self._decompress_tensor(v_caches[i], metas[i]))

        return out

    @torch.inference_mode()
    def allocate_batch_decompress_buffer(self, v_caches: torch.Tensor, metas):
        """
        Allocate and validate output buffer for `batch_decompress`.

        Returns a tensor shaped:
        (num_blocks, num_pages, page_size, num_layers, num_kv_heads, head_dim)
        """
        if not isinstance(v_caches, torch.Tensor) or v_caches.dim() < 5:
            raise ValueError(
                "v_caches must have shape (num_blocks, num_pages, page_size, num_layers, ..., head_dim/packed_head_dim)"
            )

        num_blocks = v_caches.size(0)
        if isinstance(metas, QuantMeta):
            metas = [metas] * num_blocks

        if len(metas) != num_blocks:
            raise ValueError(f"len(metas)={len(metas)} must equal num_blocks={num_blocks}")
        if any(not isinstance(m, QuantMeta) for m in metas):
            raise TypeError("each meta in metas must be QuantMeta")

        out_shape = self._infer_batch_output_shape(v_caches, metas)
        out_dtype = self._select_batch_output_dtype(v_caches, metas)
        return torch.empty(out_shape, device=v_caches.device, dtype=out_dtype)

    def _infer_batch_output_shape(self, x_batch: torch.Tensor, side_metas):
        if not side_metas:
            raise ValueError("side_metas must be non-empty")

        out_shape = None
        for meta in side_metas:
            if self.bits == 4:
                if meta.last_dim is None:
                    raise ValueError("INT4 decompress requires last_dim in meta.")
                candidate = (*x_batch.shape[:-1], meta.last_dim)
            else:
                candidate = tuple(x_batch.shape)

            if out_shape is None:
                out_shape = candidate
            elif out_shape != candidate:
                raise ValueError(
                    "inconsistent output shapes in batch_decompress: "
                    f"{out_shape} vs {candidate}"
                )

        return out_shape

    def _select_batch_output_dtype(self, v_batch, metas):
        out_dtype = metas[0].dtype if metas else v_batch.dtype
        for m in metas:
            if m.dtype != out_dtype:
                raise ValueError(
                    "inconsistent output dtype in batch_decompress: "
                    f"{out_dtype} vs {m.dtype}"
                )
        return out_dtype

    def _decompress_tensor(self, x, meta: QuantMeta):
        scale = meta.scale.to(x.device, torch.float32)

        if self.bits == 8:
            out = x.float() * scale
            return out.to(meta.dtype)

        if meta.last_dim is None:
            raise ValueError("INT4 decompress requires last_dim in meta.")
        out = int4_ext.dequant_unpack_int4(x.contiguous(), scale.contiguous(), meta.last_dim)
        return out.to(meta.dtype)
