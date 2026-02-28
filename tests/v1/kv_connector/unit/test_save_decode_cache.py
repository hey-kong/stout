# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

pytest.importorskip("lmcache")

from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.vllm_v1_adapter import (
    ReqMeta,
    RequestTracker,
)


def test_decode_phase_cache_not_saved_when_disabled() -> None:
    tracker = RequestTracker(
        req_id="req-1",
        prompt_len=4,
        token_ids=[1, 2, 3, 4, 5],
        allocated_block_ids=[0],
        num_saved_tokens=4,
        is_decode_phase=True,
    )

    req_meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=8,
        lmcache_chunk_size=1,
        save_decode_cache=False,
    )

    assert req_meta is None
    assert tracker.num_saved_tokens == 4


def test_request_override_can_save_decode_cache() -> None:
    tracker = RequestTracker(
        req_id="req-2",
        prompt_len=4,
        token_ids=[1, 2, 3, 4, 5],
        allocated_block_ids=[0],
        num_saved_tokens=4,
        is_decode_phase=True,
        request_configs={"lmcache.save_decode_cache": True},
    )

    req_meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=8,
        lmcache_chunk_size=1,
        save_decode_cache=False,
    )

    assert req_meta is not None
    assert req_meta.save_spec is not None
    assert req_meta.save_spec.can_save
    assert tracker.num_saved_tokens == 5
