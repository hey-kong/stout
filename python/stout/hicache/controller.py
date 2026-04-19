from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, NamedTuple, cast

import torch
import torch.distributed as dist
from stout.core import get_global_ctx
from stout.utils import init_logger

if TYPE_CHECKING:
    from stout.kvcache import BaseCacheHandle, BasePrefixCache
    from stout.kvcache.hiradix_cache import HiRadixPrefixCache
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
    k_src_list: List[torch.Tensor]
    v_src_list: List[torch.Tensor]
    cuda_list: List[torch.Tensor]


class Ack(NamedTuple):
    ack_id: int
    handles: List[BaseCacheHandle]
    num_tokens: int
    start_event: torch.Event
    finish_event: torch.Event


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
        self.load_split_v_stream = torch.cuda.Stream()
        self.write_stream = torch.cuda.Stream()
        self.load_stream_ctx = torch.cuda.stream(self.load_stream)
        self.load_split_v_stream_ctx = torch.cuda.stream(self.load_split_v_stream)
        self.write_stream_ctx = torch.cuda.stream(self.write_stream)
        self.num_layers, _, _, num_kv_heads, head_dim = cuda_kv[0].shape
        self.device = cuda_kv[0].device
        self.page_size = config.page_size
        item_bytes = cuda_kv[0].element_size()
        storage_shape = (-1, num_kv_heads * head_dim)
        # [num_pages, page_size, num_layers, num_kv_heads, head_dim]
        self._cuda_page = [t.permute(1, 2, 0, 3, 4) for t in cuda_kv]
        self._host_page = [t.permute(1, 2, 0, 3, 4) for t in host_kv]
        self._cuda_v_storage = cuda_kv[1]
        # 2D list of tensors with shape [num_tokens, num_kv_heads * head_dim]
        self._cuda_kv = [[t.view(storage_shape) for t in kv] for kv in cuda_kv]
        self._host_kv = [[t.view(storage_shape) for t in kv] for kv in host_kv]
        del cuda_kv, host_kv  # free original references to avoid confusion
        self._cuda_stride_bytes = self._cuda_kv[0][0].stride(0) * item_bytes
        self._host_stride_bytes = self._host_kv[0][0].stride(0) * item_bytes
        self._element_bytes = self._cuda_kv[0][0].shape[-1] * item_bytes
        self._cuda_k_ptrs = _make_ptrs(self._cuda_kv[0], self.device)
        self._cuda_v_ptrs = _make_ptrs(self._cuda_kv[1], self.device)
        self._host_k_ptrs = _make_ptrs(self._host_kv[0], self.device)
        self._host_v_ptrs = _make_ptrs(self._host_kv[1], self.device)
        self._external_v_pool = getattr(get_global_ctx(), "external_v_cache", None)

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

    def load_pages_split_kv(
            self,
            host_k_indices: torch.Tensor,
            external_v_indices: torch.Tensor,
            cuda_indices: torch.Tensor,
    ) -> None:
        assert self._external_v_pool is not None
        cuda_indices_cpu = cuda_indices.cpu() if cuda_indices.is_cuda else cuda_indices
        num_pages = len(cuda_indices) // self.page_size
        host_k_pages = torch.empty(num_pages, device="cpu", dtype=torch.int64)
        src_external_pages = torch.empty(num_pages, device="cpu", dtype=torch.int64)
        dst_cuda_pages = torch.empty(num_pages, device="cpu", dtype=torch.int64)

        for i in range(num_pages):
            host_k_page = int(host_k_indices[i * self.page_size].item()) // self.page_size
            external_v_page = int(external_v_indices[i * self.page_size].item()) // self.page_size
            cuda_page = int(cuda_indices_cpu[i * self.page_size].item()) // self.page_size

            host_k_pages[i] = host_k_page
            src_external_pages[i] = external_v_page
            dst_cuda_pages[i] = cuda_page

        for i in range(num_pages):
            self._cuda_page[0][int(dst_cuda_pages[i].item())].copy_(
                self._host_page[0][int(host_k_pages[i].item())],
                non_blocking=True,
            )

        with self.load_split_v_stream_ctx:
            self._external_v_pool.load_pages_to(
                dst_v=self._cuda_v_storage,
                src_pages=src_external_pages,
                dst_pages=dst_cuda_pages,
            )

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
        self.load_queue_split_kv: List[Transaction] = []
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
        k_list, v_list, v_from_external = self.hiradix_cache.set_cuda(host_handle, cuda_indices)
        self.hiradix_cache.lock_handle(host_handle, unlock=False)
        self.hiradix_cache.lock_handle(cuda_handle, unlock=True)
        offset = 0
        for k_src, v_src, use_external_v in zip(k_list, v_list, v_from_external):
            length = len(k_src)
            cuda_dst = cuda_indices[offset: offset + length]
            offset += length
            tx = Transaction(host_handle, [k_src], [v_src], [cuda_dst])
            if use_external_v:
                self.load_queue_split_kv.append(tx)
            else:
                self.load_queue.append(tx)

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
        self.write_queue.append(Transaction(cuda_handle, [host_indices], [host_indices], cuda_list))
        self.start_write()  # do not batch write for now

    def start_load(self) -> None:
        if not self.load_queue and not self.load_queue_split_kv:
            return self.cuda_pool.set_hicache_counter(None)
        self.ring_index = (self.ring_index + 1) % RING_SIZE
        counter = self.counter_ring_buffer[self.ring_index]
        counter.use_layerwise = self.use_layerwise
        self.cuda_pool.set_hicache_counter(counter)
        num_tokens = 0
        current_stream = torch.cuda.current_stream()
        counter.start_event.record(self.load_stream)
        with self.load_stream_ctx:
            self.load_stream.wait_stream(current_stream)
            if self.load_queue:
                host_indices, _, cuda_indices = self._merge_transactions(self.load_queue)
                num_tokens += len(host_indices)
                if self.use_layerwise:
                    for i in range(self.num_layers):
                        self.load_one(host_indices, cuda_indices, i)
                        counter.events[i].record(self.load_stream)
                elif self.pagewise_load:
                    self.load_pages(host_indices=host_indices, cuda_indices=cuda_indices)
                else:
                    self.load_all(host_indices=host_indices, cuda_indices=cuda_indices)
                host_indices.record_stream(self.load_stream)
                cuda_indices.record_stream(self.load_stream)

            if self.load_queue_split_kv:
                host_k_indices, external_v_indices, cuda_indices = self._merge_transactions_split_kv(
                    self.load_queue_split_kv
                )
                num_tokens += len(host_k_indices)
                assert self.pagewise_load, (
                    "External-V reload requires pagewise_load=True to keep page-granularity transfers"
                )
                logger.info_rank0(
                    "HiCache Load split-KV: K pages from DRAM(host), V pages from external V cache(HBM)"
                )
                self.load_pages_split_kv(
                    host_k_indices=host_k_indices,
                    external_v_indices=external_v_indices,
                    cuda_indices=cuda_indices,
                )
                self.load_stream.wait_stream(self.load_split_v_stream)
            counter.finish_event.record(self.load_stream)
        self.load_queue.clear()
        self.load_queue_split_kv.clear()
        ack_id = self._allocate_ack_id()
        self.ack_load_queue.append(Ack(ack_id, [], num_tokens, counter.start_event, counter.finish_event))
        logger.info_rank0(f"HiCache Load  [{ack_id}]: {num_tokens:>5} tokens")

    def start_write(self) -> None:
        if not self.write_queue:
            return
        handles = [tx.handle for tx in self.write_queue]
        host_indices, _, cuda_indices = self._merge_transactions(self.write_queue)
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
        host_indices.record_stream(self.write_stream)
        cuda_indices.record_stream(self.write_stream)
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

        # Compress external V pages on the write stream to keep it off the
        # inference-critical compute stream.
        if self.hiradix_cache.has_pending_external_v_sync():
            current_stream = torch.cuda.current_stream()
            with self.write_stream_ctx:
                self.write_stream.wait_stream(current_stream)
                self.hiradix_cache.flush_external_v_sync()

    def _merge_transactions(self, txs: List[Transaction]):
        assert len(txs) > 0
        host_list: List[torch.Tensor] = []
        v_list: List[torch.Tensor] = []
        cuda_list: List[torch.Tensor] = []
        for _, k_values, v_values, cuda_values in txs:
            host_list.extend(k_values)
            v_list.extend(v_values)
            cuda_list.extend(cuda_values)
        host_indices, v_indices, cuda_indices = torch.cat(host_list), torch.cat(v_list), torch.cat(cuda_list)
        return (
            host_indices.to(self.device, non_blocking=True),
            v_indices.to(self.device, non_blocking=True),
            cuda_indices,
        )

    def _merge_transactions_split_kv(self, txs: List[Transaction]):
        assert len(txs) > 0
        host_list: List[torch.Tensor] = []
        v_list: List[torch.Tensor] = []
        cuda_list: List[torch.Tensor] = []
        for _, k_values, v_values, cuda_values in txs:
            host_list.extend(k_values)
            v_list.extend(v_values)
            cuda_list.extend(cuda_values)
        return torch.cat(host_list), torch.cat(v_list), torch.cat(cuda_list)

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
        dur = ack.start_event.elapsed_time(ack.finish_event)
        bandwidth = (self.token_bytes * ack.num_tokens / (1024 ** 3)) / (dur / 1000)
        logger.info(
            f"HiCache {stage} [{ack.ack_id}]: {ack.num_tokens:>5} tokens: "
            f"duration = {dur:>5.2f} ms, bandwidth = {bandwidth:>5.2f} GB/s"
        )


# NOTE: skip the annoying type checking here...
def _create_event(enable_timing: bool = False) -> torch.Event:
    return torch.cuda.Event(enable_timing=enable_timing)  # type: ignore


def _make_ptrs(ts: List[torch.Tensor], device: torch.device):
    return torch.tensor([t.data_ptr() for t in ts], device=device, dtype=torch.uint64)
