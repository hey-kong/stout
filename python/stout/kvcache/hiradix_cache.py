from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import torch
from stout.core import get_global_ctx
from stout.utils import align_down, init_logger

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo

KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]

logger = init_logger(__name__)


class HiRadixTreeNode:
    counter: int = 0

    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: Dict[Any, HiRadixTreeNode] = {}
        self._parent: HiRadixTreeNode | None = None
        self.ref_count: int = 0
        self.uuid = HiRadixTreeNode.counter
        HiRadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()

        # these fields should be updated later
        self._key: torch.Tensor
        self._cuda_value: torch.Tensor | None = None
        self._cuda_v_value: torch.Tensor | None = None
        self._host_value: torch.Tensor | None = None
        self._length: int
        self.need_external_v_sync: bool = False

    def on_cuda_only(self) -> bool:
        return self._cuda_value is not None and self._host_value is None

    def on_host_only(self) -> bool:
        return self._cuda_value is None and self._host_value is not None

    def has_cuda_value(self) -> bool:
        return self._cuda_value is not None

    def has_host_value(self) -> bool:
        return self._host_value is not None

    def is_ghost(self) -> bool:
        return self._cuda_value is None and self._host_value is None

    def set_key_value(
        self,
        key: torch.Tensor,
        cuda_value: torch.Tensor | None,
        cuda_v_value: torch.Tensor | None = None,
        host_value: torch.Tensor | None = None,
    ) -> None:
        self._key = key
        self._cuda_value = cuda_value
        self._cuda_v_value = cuda_v_value
        self._host_value = host_value
        self._length = len(key)
        assert self._length > 0, "Node length must be greater than 0"

    def set_parent(self, parent: HiRadixTreeNode) -> None:
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> HiRadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def cuda_value(self) -> torch.Tensor:
        assert self._cuda_value is not None
        return self._cuda_value

    @property
    def host_value(self) -> torch.Tensor:
        assert self._host_value is not None
        return self._host_value

    @cuda_value.setter
    def cuda_value(self, value: torch.Tensor | None) -> None:
        if value is not None:
            assert self._cuda_value is None and len(value) == self.length
        self._cuda_value = value

    @host_value.setter
    def host_value(self, value: torch.Tensor | None) -> None:
        if value is not None:
            assert self._host_value is None and len(value) == self.length
        self._host_value = value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf_device(self) -> bool:
        return all(c._cuda_value is None and c.is_leaf_device() for c in self.children.values())

    def is_leaf_host(self) -> bool:
        return all(c._host_value is None and c.is_leaf_host() for c in self.children.values())

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        from stout.kernel import fast_compare_key

        # compare key and input_ids, find the first diff
        return fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> HiRadixTreeNode:
        assert 0 < pos < self.length
        parent = self.parent

        new_node = HiRadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(
            self._key[:pos],
            _maybe_slice(self._cuda_value, slice(0, pos)),
            _maybe_slice(self._cuda_v_value, slice(0, pos)),
            _maybe_slice(self._host_value, slice(0, pos)),
        )
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count
        self.set_key_value(
            self._key[pos:],
            _maybe_slice(self._cuda_value, slice(pos, None)),
            _maybe_slice(self._cuda_v_value, slice(pos, None)),
            _maybe_slice(self._host_value, slice(pos, None)),
        )
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: HiRadixTreeNode) -> bool:
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class HiRadixCacheHandle(BaseCacheHandle):
    node: HiRadixTreeNode

    def get_matched_indices(self) -> torch.Tensor:
        node = self.node
        value_list: List[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.cuda_value)
            node = node.parent
        value_list.reverse()
        return torch.cat(value_list)


class HiRadixPrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.page_size = get_global_ctx().page_size
        self.key_fn = _get_key_fn(self.page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = HiRadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # root is always protected
        self.host_size = 0
        self.ghost_size = 0
        self.ghost_heap: List[Tuple[int, int, HiRadixTreeNode]] = []
        ctx = get_global_ctx()
        self._kv_pool = ctx.kv_cache
        self._external_v_pool = getattr(ctx, "external_v_cache", None)
        self._external_v_free_slots: torch.Tensor | None = None
        self._pending_v_sync_nodes: set[HiRadixTreeNode] = set()
        if self._external_v_pool is not None:
            num_v_pages = self._external_v_pool.get_v_storage().shape[1]
            self._external_v_free_slots = (
                torch.arange(num_v_pages, dtype=torch.int32, device=self.device) * self.page_size
            )

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, HiRadixCacheHandle)
        node = handle.node
        assert node.is_root() or not node.on_host_only()
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        host_node, host_prefix_len = self._tree_walk(input_ids)
        while not host_node.is_root() and host_node.is_ghost():
            host_prefix_len -= host_node.length
            host_node = host_node.parent
        cuda_node, cuda_prefix_len = host_node, host_prefix_len
        while not cuda_node.is_root() and not cuda_node.has_cuda_value():
            cuda_prefix_len -= cuda_node.length
            cuda_node = cuda_node.parent
        return MatchResult(
            cuda_handle=HiRadixCacheHandle(cuda_prefix_len, cuda_node),
            host_handle=HiRadixCacheHandle(host_prefix_len, host_node),
        )

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        host_node, host_prefix_len = self._tree_walk(input_ids)
        has_ghost_match = False
        ghost_match_len = 0
        while not host_node.is_root() and host_node.is_ghost():
            has_ghost_match = True
            ghost_match_len += host_node.length
            host_prefix_len -= host_node.length
            host_node = host_node.parent
        cuda_node, cuda_prefix_len = host_node, host_prefix_len
        while not cuda_node.is_root() and not cuda_node.has_cuda_value():
            cuda_prefix_len -= cuda_node.length
            cuda_node = cuda_node.parent
        self.evictable_size += host_prefix_len - cuda_prefix_len
        updated_indices = indices[cuda_prefix_len:host_prefix_len].clone()
        node = host_node
        while not node.is_root() and node.on_host_only():
            node.cuda_value = updated_indices[-node.length :]
            node.need_external_v_sync = True
            self._pending_v_sync_nodes.add(node)
            updated_indices = updated_indices[: -node.length]
            node = node.parent
        assert len(updated_indices) == 0
        if host_prefix_len != insert_len:  # NOTE: prefix_len < insert_len
            new_node = HiRadixTreeNode(self.key_fn)
            new_node.set_key_value(
                input_ids[host_prefix_len:],
                indices[host_prefix_len:].clone(),
            )
            new_node.set_parent(host_node)
            if has_ghost_match:
                sync_len = min(ghost_match_len, new_node.length)
                if sync_len < new_node.length:
                    ghost_node = new_node.split_at(sync_len)
                    ghost_node.need_external_v_sync = True
                    self._pending_v_sync_nodes.add(ghost_node)
                else:
                    new_node.need_external_v_sync = True
                    self._pending_v_sync_nodes.add(new_node)
            self.evictable_size += new_node.length
            host_node = new_node
        return InsertResult(cuda_prefix_len, HiRadixCacheHandle(insert_len, host_node))

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        leave_nodes = self._collect_leave_nodes_for_evict(is_host=False)
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert len(leave_nodes) > 0, "Not enough evictable nodes"
            node = heapq.heappop(leave_nodes)
            evicted_size += node.length
            evicted_indices.append(node.cuda_value)
            self.evictable_size -= node.length
            parent = node.parent
            # Keep external V slots on device-side eviction. They are released
            # only when host-side data is evicted.
            if node.on_cuda_only():  # no backup on host, remove the node
                del parent.children[self.key_fn(node._key)]
            else:  # evict device part, but keep host backup
                node.cuda_value = None
            # NOTE: root is always protected, so won't be evicted
            if parent.ref_count == 0 and parent.is_leaf_device():
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)

    def try_evict_host(self, size: int) -> List[torch.Tensor]:
        if size == 0:
            return []

        leave_nodes = self._collect_leave_nodes_for_evict(is_host=True)
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size and leave_nodes:
            node = heapq.heappop(leave_nodes)
            if not node.on_host_only():  # still has device backup, skip eviction
                continue

            evicted_size += node.length
            evicted_indices.append(node.host_value)
            # Release external V slots only when host-side data is evicted.
            self._free_external_v(node)
            node.host_value = None
            self.host_size -= node.length
            self.ghost_size += node.length
            heapq.heappush(self.ghost_heap, (node.timestamp, node.uuid, node))
            self._trim_ghost_nodes()

        return evicted_indices

    def get_writable_length(self, handle: BaseCacheHandle) -> int:
        assert isinstance(handle, HiRadixCacheHandle)
        node = handle.node
        needed_len = 0
        while not node.is_root() and node.on_cuda_only():
            needed_len += node.length
            node = node.parent
        return needed_len

    def set_host(self, handle: BaseCacheHandle, indices: torch.Tensor) -> List[torch.Tensor]:
        assert isinstance(handle, HiRadixCacheHandle)
        node = handle.node
        result: List[torch.Tensor] = []
        while not node.is_root() and node.on_cuda_only():
            node.host_value = indices[-node.length :]
            self.host_size += node.length
            indices = indices[: -node.length]
            result.append(node.cuda_value)
            node = node.parent
        assert len(indices) == 0
        result.reverse()
        return result

    def set_cuda(
        self, handle: BaseCacheHandle, indices: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[bool]]:
        assert isinstance(handle, HiRadixCacheHandle)
        node = handle.node
        k_result: List[torch.Tensor] = []
        v_result: List[torch.Tensor] = []
        v_from_external: List[bool] = []
        self.evictable_size += node.length  # update evictable size first
        while not node.is_root() and node.on_host_only():
            assert node.ref_count == 0
            node.cuda_value = indices[-node.length :]
            node.need_external_v_sync = True
            self._pending_v_sync_nodes.add(node)
            indices = indices[: -node.length]
            k_result.append(node.host_value)
            if node._cuda_v_value is not None:
                v_result.append(node._cuda_v_value)
                v_from_external.append(True)
            else:
                v_result.append(node.host_value)
                v_from_external.append(False)
            node = node.parent
        assert len(indices) == 0
        k_result.reverse()
        v_result.reverse()
        v_from_external.reverse()
        return k_result, v_result, v_from_external

    def reset(self) -> None:
        raise NotImplementedError("HiRadixPrefixCache.reset is not implemented")

    def has_pending_external_v_sync(self) -> bool:
        return len(self._pending_v_sync_nodes) > 0

    def flush_external_v_sync(self) -> None:
        self._flush_external_v_sync()

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=self.evictable_size, protected_size=self.protected_size)

    def check_integrity(self) -> None:
        pass

    def _collect_leave_nodes_for_evict(self, is_host: bool) -> List[HiRadixTreeNode]:
        nodes: List[HiRadixTreeNode] = list(self.root_node.children.values())
        leave_nodes: List[HiRadixTreeNode] = []

        fn = HiRadixTreeNode.is_leaf_host if is_host else HiRadixTreeNode.is_leaf_device
        while len(nodes) > 0:
            node = nodes.pop()
            has_value = node._host_value is not None if is_host else node._cuda_value is not None
            if has_value and fn(node) and node.ref_count == 0:
                leave_nodes.append(node)
            for child in node.children.values():
                nodes.append(child)
        return leave_nodes

    def _trim_ghost_nodes(self) -> None:
        while self.ghost_size > self.host_size and self.ghost_heap:
            _, _, node = heapq.heappop(self.ghost_heap)
            if (
                node.is_root()
                or node.ref_count > 0
                or not node.is_ghost()
                or not node.is_leaf()
            ):
                continue

            parent = node.parent
            node_key = self.key_fn(node._key)
            if parent.children.get(node_key) is not node:
                continue
            del parent.children[node_key]
            self.ghost_size -= node.length
            if (
                not parent.is_root()
                and parent.ref_count == 0
                and parent.is_ghost()
                and parent.is_leaf()
            ):
                heapq.heappush(self.ghost_heap, (parent.timestamp, parent.uuid, parent))

    def _tree_walk(self, input_ids: torch.Tensor) -> Tuple[HiRadixTreeNode, int]:
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child_node is None:
                return node, prefix_len
            node = child_node  # walk to child node

            # NOTE: at least 1 page is matched, so match_len >= page_size
            match_len = node.get_match_len(input_ids[prefix_len:])
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len

            # need to split the node if not fully matched
            if match_len != node.length:
                node = node.split_at(match_len)
                return node, prefix_len

            # update timestamp for accessed node
            node.timestamp = tic

        return node, prefix_len

    def _allocate_external_v_indices(self, length: int) -> torch.Tensor | None:
        if self._external_v_free_slots is None:
            return None
        needed_pages = length // self.page_size
        if needed_pages == 0:
            return self.empty_tensor
        if needed_pages > len(self._external_v_free_slots):
            self._evict_external_v(needed_pages - len(self._external_v_free_slots))
        if needed_pages > len(self._external_v_free_slots):
            return None
        allocated = self._external_v_free_slots[:needed_pages]
        self._external_v_free_slots = self._external_v_free_slots[needed_pages:]
        if self.page_size == 1:
            return allocated
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (allocated.unsqueeze(1) + offsets).flatten()

    def _collect_nodes_with_cuda_v(self) -> List[HiRadixTreeNode]:
        nodes: List[HiRadixTreeNode] = list(self.root_node.children.values())
        candidates: List[HiRadixTreeNode] = []
        while nodes:
            node = nodes.pop()
            if node._cuda_v_value is not None:
                candidates.append(node)
            nodes.extend(node.children.values())
        return candidates

    def _evict_external_v(self, needed_pages: int) -> None:
        if needed_pages <= 0:
            return
        candidates = self._collect_nodes_with_cuda_v()
        # Evict the least recently used nodes first.
        candidates.sort(key=lambda node: node.timestamp)
        freed_pages = 0
        for node in candidates:
            if node._cuda_v_value is None:
                continue
            freed_pages += len(node._cuda_v_value) // self.page_size
            self._free_external_v(node)
            if freed_pages >= needed_pages:
                break

    def _free_external_v(self, node: HiRadixTreeNode) -> None:
        if self._external_v_free_slots is None or node._cuda_v_value is None:
            node._cuda_v_value = None
            return
        self._external_v_free_slots = torch.cat(
            [self._external_v_free_slots, node._cuda_v_value[:: self.page_size]]
        )
        node._cuda_v_value = None

    def _sync_external_v_cache(self, node: HiRadixTreeNode) -> bool:
        if self._external_v_pool is None or node._cuda_value is None:
            return False

        # Reuse existing external V slots if they already match current node length.
        # This avoids unnecessary free/re-allocate for repeated sync on the same node.
        cuda_v_indices = node._cuda_v_value
        if cuda_v_indices is None:
            cuda_v_indices = self._allocate_external_v_indices(node.length)
            if cuda_v_indices is None:
                return False
            node._cuda_v_value = cuda_v_indices
        else:
            assert len(cuda_v_indices) == node.length
            return True

        if len(cuda_v_indices) == 0:
            return True
        assert node.length % self.page_size == 0
        src_pages = (node._cuda_value[:: self.page_size] // self.page_size).to(torch.int64)
        dst_pages = (cuda_v_indices[:: self.page_size] // self.page_size).to(torch.int64)
        src_v = self._kv_pool.get_kv_storage()[1]
        self._external_v_pool.store_pages_from(src_v=src_v, src_pages=src_pages, dst_pages=dst_pages)
        return True

    def _flush_external_v_sync(self) -> None:
        if not self._pending_v_sync_nodes:
            return
        remaining_nodes: set[HiRadixTreeNode] = set()
        for node in self._pending_v_sync_nodes:
            if not node.need_external_v_sync:
                continue
            if node._cuda_value is None:
                remaining_nodes.add(node)
                continue
            if not self._sync_external_v_cache(node):
                remaining_nodes.add(node)
                continue
            node.need_external_v_sync = False
        self._pending_v_sync_nodes = remaining_nodes


def _get_key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())


def _maybe_slice(t: torch.Tensor | None, s) -> torch.Tensor | None:
    return t[s] if t is not None else None
