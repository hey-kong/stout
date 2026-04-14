from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, NamedTuple, cast

import torch
import torch.distributed as dist
from stout.core import get_global_ctx
from stout.quant import QuantizedCompressor
from stout.utils import init_logger

if TYPE_CHECKING:
    from stout.kvcache import BaseCacheHandle, BasePrefixCache
    from stout.kvcache.hiradix_cache import HiRadixPrefixCache, HiRadixTreeNode
    from stout.scheduler import SchedulerConfig

logger = init_logger(__name__)


@dataclass
class HiCacheCounter:
    num_layers: int
    use_layerwise: bool = True
    start_event: torch.Event = field(init=False)
    finish_event: torch.Event = field(init=False)
    events: List[torch.Event] = field(init=False)

    def __post_init__(self):
        self.start_event = _create_event(enable_timing=True)
        self.finish_event = _create_event(enable_timing=True)
        self.events = [_create_event() for _ in range(self.num_layers)]

    def wait(self, layer_id: int) -> None:
        current_stream = torch.cuda.current_stream()
        if self.use_layerwise:
            current_stream.wait_event(self.events[layer_id])
        else:
            current_stream.wait_event(self.finish_event)


class Transaction(NamedTuple):
    handle: BaseCacheHandle
    host_list: List[torch.Tensor]
    cuda_list: List[torch.Tensor]
    node_list: List["HiRadixTreeNode"]


class Ack(NamedTuple):
    ack_id: int
    handles: List[BaseCacheHandle]
    num_tokens: int
    start_event: torch.Event
    finish_event: torch.Event
    extra_tokens: int = 0
    extra_start_event: torch.Event | None = None
    extra_finish_event: torch.Event | None = None


RING_SIZE = 3  # 3 is enough and safe
RESET_ACK_THRESHOLD = 512


class HiCacheTransferMixin:
    def __init__(
            self,
            cuda_kv: List[torch.Tensor],
            host_kv: List[torch.Tensor],
            config: SchedulerConfig,
    ) -> None:
        self.load_stream = torch.cuda.Stream()
        self.decompress_stream = torch.cuda.Stream()
        self.compress_stream = torch.cuda.Stream()
        self.write_stream = torch.cuda.Stream()
        self.load_stream_ctx = torch.cuda.stream(self.load_stream)
        self.decompress_stream_ctx = torch.cuda.stream(self.decompress_stream)
        self.compress_stream_ctx = torch.cuda.stream(self.compress_stream)
        self.write_stream_ctx = torch.cuda.stream(self.write_stream)
        self.num_layers, _, _, num_kv_heads, head_dim = cuda_kv[0].shape
        self.device = cuda_kv[0].device
        self.page_size = config.page_size
        item_bytes = cuda_kv[0].element_size()
        storage_shape = (-1, num_kv_heads * head_dim)
        # [num_pages, page_size, num_layers, num_kv_heads, head_dim]
        self._cuda_page = [t.permute(1, 2, 0, 3, 4) for t in cuda_kv]
        self._host_page = [t.permute(1, 2, 0, 3, 4) for t in host_kv]
        # 2D list of tensors with shape [num_tokens, num_kv_heads * head_dim]
        self._cuda_kv = [[t.view(storage_shape) for t in kv] for kv in cuda_kv]
        self._host_kv = [[t.view(storage_shape) for t in kv] for kv in host_kv]
        del cuda_kv, host_kv  # free original references to avoid confusion
        self._cuda_stride_bytes = self._cuda_kv[0][0].stride(0) * item_bytes
        self._host_stride_bytes = self._host_kv[0][0].stride(0) * item_bytes
        self._element_bytes = self._cuda_kv[0][0].shape[-1] * item_bytes
        self._v_token_bytes = self._element_bytes * self.num_layers
        self._cuda_k_ptrs = _make_ptrs(self._cuda_kv[0], self.device)
        self._cuda_v_ptrs = _make_ptrs(self._cuda_kv[1], self.device)
        self._host_k_ptrs = _make_ptrs(self._host_kv[0], self.device)
        self._host_v_ptrs = _make_ptrs(self._host_kv[1], self.device)
        self.quant_compressor = QuantizedCompressor(bits=4)

    def load_one(self, host_indices: torch.Tensor, cuda_indices: torch.Tensor, i: int) -> None:
        from stout.kernel import transfer_hicache_one_layer

        transfer_hicache_one_layer(
            k_cache_dst=self._cuda_kv[0][i],
            v_cache_dst=self._cuda_kv[1][i],
            indices_dst=cuda_indices,
            k_cache_src=self._host_kv[0][i],
            v_cache_src=self._host_kv[1][i],
            indices_src=host_indices,
        )

    def load_all(self, host_indices: torch.Tensor, cuda_indices: torch.Tensor) -> None:
        from stout.kernel import transfer_hicache_all_layer

        transfer_hicache_all_layer(
            k_ptr_dst=self._cuda_k_ptrs,
            v_ptr_dst=self._cuda_v_ptrs,
            indices_dst=cuda_indices,
            k_ptr_src=self._host_k_ptrs,
            v_ptr_src=self._host_v_ptrs,
            indices_src=host_indices,
            kv_cache_dst_stride_bytes=self._cuda_stride_bytes,
            kv_cache_src_stride_bytes=self._host_stride_bytes,
            element_size=self._element_bytes,
        )

    def load_pages(self, host_indices: torch.Tensor, cuda_indices: torch.Tensor) -> None:
        num_pages = len(host_indices) // self.page_size

        # fast path
        if (int(host_indices[-1].item()) == int(host_indices[0].item()) + len(host_indices) - 1
                and int(cuda_indices[-1].item()) == int(cuda_indices[0].item()) + len(cuda_indices) - 1):
            host_page_start = int(host_indices[0].item()) // self.page_size
            cuda_page_start = int(cuda_indices[0].item()) // self.page_size

            self._cuda_page[0][cuda_page_start:cuda_page_start + num_pages].copy_(
                self._host_page[0][host_page_start:host_page_start + num_pages],
                non_blocking=True,
            )
            self._cuda_page[1][cuda_page_start:cuda_page_start + num_pages].copy_(
                self._host_page[1][host_page_start:host_page_start + num_pages],
                non_blocking=True,
            )
            return

        for i in range(num_pages):
            host_page = int(host_indices[i * self.page_size].item()) // self.page_size
            cuda_page = int(cuda_indices[i * self.page_size].item()) // self.page_size

            self._cuda_page[0][cuda_page].copy_(
                self._host_page[0][host_page],
                non_blocking=True,
            )
            self._cuda_page[1][cuda_page].copy_(
                self._host_page[1][host_page],
                non_blocking=True,
            )

    def load_k_pages(self, host_indices: torch.Tensor, cuda_indices: torch.Tensor) -> None:
        num_pages = len(host_indices) // self.page_size

        # fast path
        if (int(host_indices[-1].item()) == int(host_indices[0].item()) + len(host_indices) - 1
                and int(cuda_indices[-1].item()) == int(cuda_indices[0].item()) + len(cuda_indices) - 1):
            host_page_start = int(host_indices[0].item()) // self.page_size
            cuda_page_start = int(cuda_indices[0].item()) // self.page_size

            self._cuda_page[0][cuda_page_start:cuda_page_start + num_pages].copy_(
                self._host_page[0][host_page_start:host_page_start + num_pages],
                non_blocking=True,
            )
            return

        for i in range(num_pages):
            host_page = int(host_indices[i * self.page_size].item()) // self.page_size
            cuda_page = int(cuda_indices[i * self.page_size].item()) // self.page_size

            self._cuda_page[0][cuda_page].copy_(
                self._host_page[0][host_page],
                non_blocking=True,
            )

    def load_k_all(self, host_indices: torch.Tensor, cuda_indices: torch.Tensor) -> None:
        for i in range(self.num_layers):
            src = self._host_kv[0][i][host_indices].to(self.device, non_blocking=True)
            self._cuda_kv[0][i][cuda_indices].copy_(src, non_blocking=True)

    def store_all(self, host_indices: torch.Tensor, cuda_indices: torch.Tensor) -> None:
        from stout.kernel import transfer_hicache_all_layer

        transfer_hicache_all_layer(
            k_ptr_dst=self._host_k_ptrs,
            v_ptr_dst=self._host_v_ptrs,
            indices_dst=host_indices,
            k_ptr_src=self._cuda_k_ptrs,
            v_ptr_src=self._cuda_v_ptrs,
            indices_src=cuda_indices,
            kv_cache_dst_stride_bytes=self._host_stride_bytes,
            kv_cache_src_stride_bytes=self._cuda_stride_bytes,
            element_size=self._element_bytes,
        )


class HiCacheController(HiCacheTransferMixin):
    def __init__(self, prefix_cache: BasePrefixCache, num_pages: int, config: SchedulerConfig):
        self.hiradix_cache = cast("HiRadixPrefixCache", prefix_cache)
        self.load_queue: List[Transaction] = []
        self.write_queue: List[Transaction] = []
        self.ack_load_queue: List[Ack] = []
        self.ack_write_queue: List[Ack] = []
        self.ack_cnt = 0
        self.cuda_pool = get_global_ctx().kv_cache
        self.num_layers = self.cuda_pool.num_layers
        self.use_layerwise = config.use_layerwise
        self.pagewise_load = (
            config.device_mem_layout == "page_first"
            and config.host_mem_layout == "page_first"
            and not self.use_layerwise
        )
        self.ring_index = 0
        self.counter_ring_buffer = [HiCacheCounter(self.num_layers) for _ in range(RING_SIZE)]
        self.token_bytes = self.cuda_pool.get_per_token_bytes()
        self.external_cache_budget_bytes = int(
            getattr(get_global_ctx(), "external_cache_budget_bytes", 0)
        )
        self.external_cache_used_bytes = 0
        if config.kv_offloading_size is not None:
            num_host_tokens = int(config.kv_offloading_size * (1024 ** 3) / self.token_bytes)
            num_host_pages = num_host_tokens // config.page_size
        else:
            num_host_pages = int(num_pages * config.hicache_ratio)
        num_host_tokens = num_host_pages * config.page_size
        total_bytes_gb = num_host_tokens * self.token_bytes / (1024 ** 3)
        self.free_slots = torch.arange(num_host_tokens, dtype=torch.int32, device="cpu")
        logger.info(
            f"Allocating {num_host_tokens} tokens "
            f"({total_bytes_gb:.2f} GB) for host memory pool"
        )
        self.host_pool = self.cuda_pool.create_host_pool(num_host_pages, config.host_mem_layout)
        if self.external_cache_budget_bytes > 0:
            logger.info(
                "External cache budget enabled: "
                f"{self.external_cache_budget_bytes / (1024 ** 3):.2f} GB"
            )

        # tuple of kv, shape [num_layers, num_pages, num_kv_heads, head_dim]
        super().__init__(
            cuda_kv=list(self.cuda_pool.get_kv_storage()),
            host_kv=list(self.host_pool.get_kv_storage()),
            config=config,
        )

    def prepare_load(
            self,
            host_handle: BaseCacheHandle,
            cuda_handle: BaseCacheHandle,
            cuda_indices: torch.Tensor,
    ) -> None:
        # Collect host-only nodes before promoting them to cuda-backed nodes.
        # `set_cuda` mutates each traversed node from host-only -> mixed host+cuda,
        # so collecting afterwards would return an empty list and break alignment.
        node_list = self._collect_host_only_nodes(host_handle)
        host_list = self.hiradix_cache.set_cuda(host_handle, cuda_indices)
        assert len(host_list) == len(node_list)
        cuda_list: List[torch.Tensor] = []
        offset = 0
        for host_indices in host_list:
            length = len(host_indices)
            cuda_list.append(cuda_indices[offset: offset + length])
            offset += length
        assert offset == len(cuda_indices)
        for node in node_list:
            if node._cuda_v_value is None and node._host_value is not None:
                start_event = _create_event(enable_timing=True)
                finish_event = _create_event(enable_timing=True)
                current_stream = torch.cuda.current_stream()
                with self.compress_stream_ctx:
                    self.compress_stream.wait_stream(current_stream)
                    start_event.record(self.compress_stream)
                    compressed = self._compress_v_to_external(node._host_value)
                    if compressed is None:
                        continue
                    node._cuda_v_value = compressed
                    node._cuda_v_ready_event = _create_event()
                    node._cuda_v_ready_event.record(self.compress_stream)
                    finish_event.record(self.compress_stream)
                self._log_simple_transaction(
                    stage="Compress",
                    num_tokens=node.length,
                    bytes_per_token=self._v_token_bytes,
                    start_event=start_event,
                    finish_event=finish_event,
                )
        self.hiradix_cache.lock_handle(host_handle, unlock=False)
        self.hiradix_cache.lock_handle(cuda_handle, unlock=True)
        self.load_queue.append(Transaction(host_handle, host_list, cuda_list, node_list))

    def prepare_write(self, cuda_handle: BaseCacheHandle) -> None:
        needed_len = self.hiradix_cache.get_writable_length(cuda_handle)
        if needed_len < self.page_size:
            return
        host_indices = self._try_allocate_host(needed_len)
        if host_indices is None:
            return
        assert len(host_indices) == needed_len
        cuda_list = self.hiradix_cache.set_host(cuda_handle, host_indices)
        self.hiradix_cache.lock_handle(cuda_handle, unlock=False)
        self.write_queue.append(Transaction(cuda_handle, [host_indices], cuda_list, []))
        self.start_write()  # do not batch write for now

    def start_load(self) -> None:
        if not self.load_queue:
            return self.cuda_pool.set_hicache_counter(None)
        self.ring_index = (self.ring_index + 1) % RING_SIZE
        counter = self.counter_ring_buffer[self.ring_index]
        counter.use_layerwise = self.use_layerwise
        self.cuda_pool.set_hicache_counter(counter)
        num_tokens = sum(sum(len(v) for v in tx.host_list) for tx in self.load_queue)
        current_stream = torch.cuda.current_stream()
        counter.start_event.record(self.load_stream)
        with self.load_stream_ctx:
            self.load_stream.wait_stream(current_stream)
            if self.use_layerwise:
                for i in range(self.num_layers):
                    for tx in self.load_queue:
                        for host_indices, cuda_indices, node in zip(tx.host_list, tx.cuda_list, tx.node_list):
                            if node._cuda_v_value is not None:
                                src = self._host_kv[0][i][host_indices].to(self.device, non_blocking=True)
                                self._cuda_kv[0][i][cuda_indices].copy_(src, non_blocking=True)
                            else:
                                self.load_one(host_indices.to(self.device, non_blocking=True), cuda_indices, i)
                    counter.events[i].record(self.load_stream)
            else:
                for tx in self.load_queue:
                    for host_indices, cuda_indices, node in zip(tx.host_list, tx.cuda_list, tx.node_list):
                        if node._cuda_v_value is not None:
                            if self.pagewise_load:
                                self.load_k_pages(
                                    host_indices=host_indices.to(self.device, non_blocking=True),
                                    cuda_indices=cuda_indices,
                                )
                            else:
                                self.load_k_all(
                                    host_indices=host_indices.to(self.device, non_blocking=True),
                                    cuda_indices=cuda_indices,
                                )
                        elif self.pagewise_load:
                            self.load_pages(
                                host_indices=host_indices.to(self.device, non_blocking=True),
                                cuda_indices=cuda_indices,
                            )
                        else:
                            self.load_all(
                                host_indices=host_indices.to(self.device, non_blocking=True),
                                cuda_indices=cuda_indices,
                            )
            decomp_start_event = _create_event(enable_timing=True)
            decomp_finish_event = _create_event(enable_timing=True)
            decomp_event = _create_event()
            decomp_tokens = 0
            with self.decompress_stream_ctx:
                self.decompress_stream.wait_stream(current_stream)
                decomp_start_event.record(self.decompress_stream)
                for tx in self.load_queue:
                    for _, cuda_indices, node in zip(tx.host_list, tx.cuda_list, tx.node_list):
                        if node._cuda_v_value is None:
                            continue
                        decomp_tokens += len(cuda_indices)
                        self._decompress_v_to_cuda(node, cuda_indices)
                decomp_finish_event.record(self.decompress_stream)
                decomp_event.record(self.decompress_stream)
            self.load_stream.wait_event(decomp_event)
            counter.finish_event.record(self.load_stream)

        for tx in self.load_queue:
            for host_indices, cuda_indices in zip(tx.host_list, tx.cuda_list):
                _maybe_record_stream(host_indices, self.load_stream)
                _maybe_record_stream(cuda_indices, self.load_stream)
        self.load_queue.clear()
        ack_id = self._allocate_ack_id()
        self.ack_load_queue.append(
            Ack(
                ack_id,
                [],
                num_tokens,
                counter.start_event,
                counter.finish_event,
                decomp_tokens,
                decomp_start_event,
                decomp_finish_event,
            )
        )
        logger.info_rank0(f"HiCache Load  [{ack_id}]: {num_tokens:>5} tokens")

    def start_write(self) -> None:
        if not self.write_queue:
            return
        handles = [tx.handle for tx in self.write_queue]
        host_indices, cuda_indices = self._merge_transactions(self.write_queue)
        num_tokens = len(host_indices)
        current_stream = torch.cuda.current_stream()
        start_event = _create_event(enable_timing=True)
        finish_event = _create_event(enable_timing=True)
        start_event.record(self.write_stream)
        with self.write_stream_ctx:
            self.write_stream.wait_stream(current_stream)
            self.store_all(host_indices, cuda_indices)

        # NOTE: must record stream to avoid use after free
        finish_event.record(self.write_stream)
        _maybe_record_stream(host_indices, self.write_stream)
        _maybe_record_stream(cuda_indices, self.write_stream)
        self.write_queue.clear()
        ack_id = self._allocate_ack_id()
        self.ack_write_queue.append(Ack(ack_id, handles, num_tokens, start_event, finish_event))
        logger.info_rank0(f"HiCache Write [{ack_id}]: {num_tokens:>5} tokens")

    def refresh(self, tp_cpu_group: torch.distributed.ProcessGroup) -> None:
        # NOTE: load has no side-effect (only logging), so no need to sync
        finish_count = 0
        for ack in self.ack_load_queue:
            if not ack.finish_event.query():
                break
            finish_count += 1
            self._log_transaction(ack, "Load ")
            if (
                ack.extra_tokens > 0
                and ack.extra_start_event is not None
                and ack.extra_finish_event is not None
            ):
                self._log_simple_transaction(
                    stage="Decompress",
                    num_tokens=ack.extra_tokens,
                    bytes_per_token=self._v_token_bytes,
                    start_event=ack.extra_start_event,
                    finish_event=ack.extra_finish_event,
                    ack_id=ack.ack_id,
                )
        self.ack_load_queue = self.ack_load_queue[finish_count:]

        finish_count = 0
        for ack in self.ack_write_queue:
            if not ack.finish_event.query():
                break
            finish_count += 1

        # NOTE: write must synchronize to reach consensus on the finished count
        finish_count = torch.tensor(finish_count, dtype=torch.int32, device="cpu")
        dist.all_reduce(finish_count, op=dist.ReduceOp.MIN, group=tp_cpu_group)
        finish_count = int(finish_count)

        for ack in self.ack_write_queue[:finish_count]:
            self._log_transaction(ack, "Write")
            for handle in ack.handles:
                self.hiradix_cache.lock_handle(handle, unlock=True)
        self.ack_write_queue = self.ack_write_queue[finish_count:]

    def _merge_transactions(self, txs: List[Transaction]):
        assert len(txs) > 0
        host_list: List[torch.Tensor] = []
        cuda_list: List[torch.Tensor] = []
        for _, host_values, cuda_values, _ in txs:
            host_list.extend(host_values)
            cuda_list.extend(cuda_values)
        host_indices, cuda_indices = torch.cat(host_list), torch.cat(cuda_list)
        return host_indices.to(self.device, non_blocking=True), cuda_indices

    def _collect_host_only_nodes(self, host_handle: BaseCacheHandle) -> List["HiRadixTreeNode"]:
        node = cast("HiRadixCacheHandle", host_handle).node
        nodes: List["HiRadixTreeNode"] = []
        while not node.is_root() and node.on_host_only():
            nodes.append(node)
            node = node.parent
        nodes.reverse()
        return nodes

    def _compress_v_to_external(self, host_indices: torch.Tensor):
        num_tokens = len(host_indices)
        num_pages = num_tokens // self.page_size
        with torch.inference_mode():
            v_layers = [
                self._host_kv[1][i][host_indices].to(self.device, non_blocking=True).view(
                    num_pages, self.page_size, -1
                )
                for i in range(self.num_layers)
            ]
            v_stack = torch.stack(v_layers, dim=2).view(num_pages, self.page_size, self.num_layers, -1)
            num_kv_heads = self._cuda_page[1].shape[-2]
            head_dim = self._cuda_page[1].shape[-1]
            v_cache = v_stack.view(num_pages, self.page_size, self.num_layers, num_kv_heads, head_dim)
            compressed = self.quant_compressor.compress(v_cache)
            compressed_bytes = compressed[0].numel() * compressed[0].element_size()
            if (
                self.external_cache_budget_bytes > 0
                and self.external_cache_used_bytes + compressed_bytes > self.external_cache_budget_bytes
            ):
                logger.warning(
                    "Skip external V compression due to budget limit: "
                    f"used={self.external_cache_used_bytes / (1024 ** 3):.2f} GB, "
                    f"need={compressed_bytes / (1024 ** 3):.2f} GB, "
                    f"budget={self.external_cache_budget_bytes / (1024 ** 3):.2f} GB"
                )
                return None
            self.external_cache_used_bytes += compressed_bytes
            return compressed

    def _decompress_v_to_cuda(self, node: "HiRadixTreeNode", cuda_indices: torch.Tensor) -> None:
        if node._cuda_v_ready_event is not None:
            self.decompress_stream.wait_event(node._cuda_v_ready_event)
        compressed_v, meta = node.cuda_v_value
        compressed_bytes = compressed_v.numel() * compressed_v.element_size()
        v_cache = self.quant_compressor.decompress(compressed_v, meta)
        flat_v = v_cache.view(-1, self.num_layers, v_cache.shape[-2] * v_cache.shape[-1])
        for i in range(self.num_layers):
            self._cuda_kv[1][i][cuda_indices].copy_(flat_v[:, i], non_blocking=True)
        node._cuda_v_value = None
        node._cuda_v_ready_event = None
        self.external_cache_used_bytes = max(0, self.external_cache_used_bytes - compressed_bytes)

    def _try_allocate_host(self, length: int) -> torch.Tensor | None:
        if length > len(self.free_slots):
            evicted = self.hiradix_cache.try_evict_host(length - len(self.free_slots))
            self.free_slots = torch.cat([self.free_slots] + evicted)
            if length > len(self.free_slots):  # give up if still not enough
                return None
        allocated, self.free_slots = self.free_slots[:length], self.free_slots[length:]
        return allocated

    def _allocate_counter(self) -> HiCacheCounter:
        self.ring_index = (self.ring_index + 1) % RING_SIZE
        return self.counter_ring_buffer[self.ring_index]

    def _allocate_ack_id(self) -> int:
        self.ack_cnt = (self.ack_cnt + 1) % RESET_ACK_THRESHOLD
        return self.ack_cnt

    def _log_transaction(self, ack: Ack, stage: str):
        if not ack.start_event.query() or not ack.finish_event.query():
            logger.debug(
                "Skip HiCache %s timing log for ack %s because CUDA events are not complete yet",
                stage,
                ack.ack_id,
            )
            return
        dur = ack.start_event.elapsed_time(ack.finish_event)
        bandwidth = (self.token_bytes * ack.num_tokens / (1024 ** 3)) / (dur / 1000)
        logger.info(
            f"HiCache {stage} [{ack.ack_id}]: {ack.num_tokens:>5} tokens: "
            f"duration = {dur:>5.2f} ms, bandwidth = {bandwidth:>5.2f} GB/s"
        )

    def _log_simple_transaction(
        self,
        stage: str,
        num_tokens: int,
        bytes_per_token: int,
        start_event: torch.Event,
        finish_event: torch.Event,
        ack_id: int | None = None,
    ) -> None:
        if not start_event.query() or not finish_event.query():
            logger.debug(
                "Skip HiCache %s timing log%s because CUDA events are not complete yet",
                stage,
                "" if ack_id is None else f" [ack={ack_id}]",
            )
            return
        dur = start_event.elapsed_time(finish_event)
        bandwidth = (bytes_per_token * num_tokens / (1024 ** 3)) / (dur / 1000)
        if ack_id is None:
            logger.info(
                f"HiCache {stage}: {num_tokens:>5} tokens: "
                f"duration = {dur:>5.2f} ms, bandwidth = {bandwidth:>5.2f} GB/s"
            )
            return
        logger.info(
            f"HiCache {stage} [{ack_id}]: {num_tokens:>5} tokens: "
            f"duration = {dur:>5.2f} ms, bandwidth = {bandwidth:>5.2f} GB/s"
        )


# NOTE: skip the annoying type checking here...
def _create_event(enable_timing: bool = False) -> torch.Event:
    return torch.cuda.Event(enable_timing=enable_timing)  # type: ignore


def _make_ptrs(ts: List[torch.Tensor], device: torch.device):
    return torch.tensor([t.data_ptr() for t in ts], device=device, dtype=torch.uint64)


def _maybe_record_stream(tensor: torch.Tensor, stream: torch.cuda.Stream) -> None:
    if tensor.is_cuda:
        tensor.record_stream(stream)
