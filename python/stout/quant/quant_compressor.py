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


@dataclass
class KVQuantMeta:
    k: Optional[QuantMeta] = None
    v: Optional[QuantMeta] = None


class QuantizedCompressor:
    def __init__(self, bits: int = 4):
        assert bits in (4, 8)
        self.bits = bits

    @torch.inference_mode()
    def compress(self, kv_cache: torch.Tensor, mode: str = "kv"):
        """
        kv_cache layout: (2, num_layers, block_size, num_kv_heads, head_size)
          - kv_cache[0] is K
          - kv_cache[1] is V

        mode:
          - "kv": compress both K and V
          - "k" : compress only K (V kept as-is)
          - "v" : compress only V (K kept as-is)

        Quantization is done per-layer (num_layers dimension).
        """
        assert mode in ("kv", "k", "v")
        if kv_cache.dim() < 3 or kv_cache.size(0) != 2:
            raise ValueError("kv_cache must have shape (2, num_layers, ..., head_size)")

        keys = kv_cache[0]
        values = kv_cache[1]
        ck, cv = keys, values
        meta = KVQuantMeta()

        if mode in ("kv", "k"):
            ck, meta_k = self._compress_tensor(keys, keys.dtype)
            meta.k = meta_k

        if mode in ("kv", "v"):
            cv, meta_v = self._compress_tensor(values, values.dtype)
            meta.v = meta_v

        return ck, cv, meta

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
        # tensor shape: (num_layers, ...)
        if tensor.dim() < 1:
            raise ValueError("tensor must have at least 1 dimension for layer-wise quantization")
        if tensor.dim() == 1:
            return tensor.abs()

        reduce_dims = tuple(range(1, tensor.dim()))
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
    def decompress(self, kv_cache, meta: KVQuantMeta):
        """
        Decompress a KV cache block and restore layout:
        (2, num_layers, block_size, num_kv_heads, head_size)

        kv_cache can be:
          - Tensor with shape (2, num_layers, ..., head_size/packed_head_size)
          - Tuple/List: (keys, values). This form supports mixed shapes in
            partial modes (e.g. mode="k" or mode="v").

        NOTE:
          For partial modes, do NOT stack compressed K/V before calling this
          function because packed and unpacked last dimensions differ.
          Use: decompress((ck, cv), meta)
        """
        if not isinstance(meta, KVQuantMeta):
            raise TypeError("meta must be KVQuantMeta")

        if isinstance(kv_cache, torch.Tensor):
            if kv_cache.dim() < 3 or kv_cache.size(0) != 2:
                raise ValueError(
                    "kv_cache must have shape (2, num_layers, ..., head_size/packed_head_size)"
                )
            keys, values = kv_cache[0], kv_cache[1]
        elif isinstance(kv_cache, (tuple, list)) and len(kv_cache) == 2:
            keys, values = kv_cache[0], kv_cache[1]
        else:
            raise TypeError("kv_cache must be a Tensor(2, ...) or a (keys, values) tuple/list")

        dk, dv = keys, values

        if meta.k is not None:
            dk = self._decompress_tensor(keys, meta.k)
        if meta.v is not None:
            dv = self._decompress_tensor(values, meta.v)

        return torch.stack([dk, dv], dim=0)

    @torch.inference_mode()
    def batch_decompress(self, kv_caches, metas, out: Optional[torch.Tensor] = None):
        """
        Decompress batched KV cache blocks and restore layout:
        (num_blocks, 2, num_layers, block_size, num_kv_heads, head_size)

        Args:
            kv_caches:
              - Tensor shaped (num_blocks, 2, num_layers, ..., head_size/packed_head_size)
              - Tuple/List: (keys_batch, values_batch) where each has shape
                (num_blocks, num_layers, ..., *) and * can differ between K/V
                for partial modes.
            metas: sequence of KVQuantMeta with length == num_blocks,
                   or a single KVQuantMeta to be reused for all blocks.
            out: Optional preallocated output tensor returned by
                 `allocate_batch_decompress_buffer`. If provided,
                 batch_decompress only performs decompression and write-back.
                 If None, keeps original logic by decompressing each block and
                 stacking the results.
        """
        if isinstance(kv_caches, torch.Tensor):
            if kv_caches.dim() < 4 or kv_caches.size(1) != 2:
                raise ValueError(
                    "kv_caches must have shape (num_blocks, 2, num_layers, ..., head_size/packed_head_size)"
                )
            keys_batch, values_batch = kv_caches[:, 0], kv_caches[:, 1]
        elif isinstance(kv_caches, (tuple, list)) and len(kv_caches) == 2:
            keys_batch, values_batch = kv_caches[0], kv_caches[1]
            if keys_batch.size(0) != values_batch.size(0):
                raise ValueError("keys_batch and values_batch must share the same num_blocks")
        else:
            raise TypeError(
                "kv_caches must be Tensor(num_blocks, 2, ...) or a (keys_batch, values_batch) tuple/list"
            )

        num_blocks = keys_batch.size(0)

        if out is None:
            if isinstance(metas, KVQuantMeta):
                metas = [metas] * num_blocks

            if len(metas) != num_blocks:
                raise ValueError(f"len(metas)={len(metas)} must equal num_blocks={num_blocks}")
            if any(not isinstance(m, KVQuantMeta) for m in metas):
                raise TypeError("each meta in metas must be KVQuantMeta")

            return torch.stack([
                self.decompress((keys_batch[i], values_batch[i]), metas[i])
                for i in range(num_blocks)
            ], dim=0)

        for i in range(num_blocks):
            meta_k = metas[i].k
            meta_v = metas[i].v

            if meta_k is None:
                out[i, 0].copy_(keys_batch[i])
            else:
                out[i, 0].copy_(self._decompress_tensor(keys_batch[i], meta_k))

            if meta_v is None:
                out[i, 1].copy_(values_batch[i])
            else:
                out[i, 1].copy_(self._decompress_tensor(values_batch[i], meta_v))

        return out

    @torch.inference_mode()
    def allocate_batch_decompress_buffer(self, kv_caches, metas):
        """
        Allocate and validate output buffer for `batch_decompress`.

        Returns a tensor shaped:
        (num_blocks, 2, num_layers, block_size, num_kv_heads, head_size)
        """
        if isinstance(kv_caches, torch.Tensor):
            if kv_caches.dim() < 4 or kv_caches.size(1) != 2:
                raise ValueError(
                    "kv_caches must have shape (num_blocks, 2, num_layers, ..., head_size/packed_head_size)"
                )
            keys_batch, values_batch = kv_caches[:, 0], kv_caches[:, 1]
        elif isinstance(kv_caches, (tuple, list)) and len(kv_caches) == 2:
            keys_batch, values_batch = kv_caches[0], kv_caches[1]
            if keys_batch.size(0) != values_batch.size(0):
                raise ValueError("keys_batch and values_batch must share the same num_blocks")
        else:
            raise TypeError(
                "kv_caches must be Tensor(num_blocks, 2, ...) or a (keys_batch, values_batch) tuple/list"
            )

        num_blocks = keys_batch.size(0)
        if isinstance(metas, KVQuantMeta):
            metas = [metas] * num_blocks

        if len(metas) != num_blocks:
            raise ValueError(f"len(metas)={len(metas)} must equal num_blocks={num_blocks}")
        if any(not isinstance(m, KVQuantMeta) for m in metas):
            raise TypeError("each meta in metas must be KVQuantMeta")

        meta_k = [m.k for m in metas]
        meta_v = [m.v for m in metas]

        out_k_shape = self._infer_batch_output_shape(keys_batch, meta_k)
        out_v_shape = self._infer_batch_output_shape(values_batch, meta_v)
        if out_k_shape != out_v_shape:
            raise ValueError(
                "decompressed K/V output shapes must match; got "
                f"K={out_k_shape}, V={out_v_shape}"
            )

        out_dtype = self._select_batch_output_dtype(keys_batch, values_batch, meta_k, meta_v)
        expected_shape = (num_blocks, 2, *out_k_shape[1:])
        return torch.empty(expected_shape, device=keys_batch.device, dtype=out_dtype)

    def _infer_batch_output_shape(self, x_batch: torch.Tensor, side_metas):
        if not side_metas:
            raise ValueError("side_metas must be non-empty")

        out_shape = None
        for meta in side_metas:
            if meta is None:
                candidate = tuple(x_batch.shape)
            elif self.bits == 4:
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

    def _select_batch_output_dtype(self, keys_batch, values_batch, meta_k, meta_v):
        key_dtype = meta_k[0].dtype if meta_k and meta_k[0] is not None else keys_batch.dtype
        val_dtype = meta_v[0].dtype if meta_v and meta_v[0] is not None else values_batch.dtype

        for m in meta_k:
            if (m is not None) and (m.dtype != key_dtype):
                raise ValueError(
                    "inconsistent K output dtype in batch_decompress: "
                    f"{key_dtype} vs {m.dtype}"
                )
        for m in meta_v:
            if (m is not None) and (m.dtype != val_dtype):
                raise ValueError(
                    "inconsistent V output dtype in batch_decompress: "
                    f"{val_dtype} vs {m.dtype}"
                )

        if key_dtype != val_dtype:
            raise ValueError(
                "K/V output dtype mismatch in batch_decompress: "
                f"{key_dtype} vs {val_dtype}"
            )
        return key_dtype

    def _decompress_tensor(self, x, meta: QuantMeta):
        scale = meta.scale.to(x.device, torch.float32)

        if self.bits == 8:
            out = x.float() * scale
            return out.to(meta.dtype)

        if meta.last_dim is None:
            raise ValueError("INT4 decompress requires last_dim in meta.")
        out = int4_ext.dequant_unpack_int4(x.contiguous(), scale.contiguous(), meta.last_dim)
        return out.to(meta.dtype)
