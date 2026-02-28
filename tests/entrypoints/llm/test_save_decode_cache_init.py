# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.kv_transfer import KVTransferConfig
from vllm.entrypoints.llm import LLM


class _DummyEngine:
    pass


def test_save_decode_cache_as_top_level_llm_kwarg(monkeypatch):
    captured = {}

    def fake_from_engine_args(*, engine_args, usage_context):
        captured["engine_args"] = engine_args
        captured["usage_context"] = usage_context
        return _DummyEngine()

    monkeypatch.setattr(
        "vllm.entrypoints.llm.LLMEngine.from_engine_args",
        fake_from_engine_args,
    )
    monkeypatch.setattr("vllm.entrypoints.llm.log_non_default_args", lambda *_: None)

    LLM(model="facebook/opt-125m", save_decode_cache=True)

    kv_transfer_config = captured["engine_args"].kv_transfer_config
    assert isinstance(kv_transfer_config, KVTransferConfig)
    assert kv_transfer_config.save_decode_cache is True


def test_save_decode_cache_overrides_kv_transfer_config_dict(monkeypatch):
    captured = {}

    def fake_from_engine_args(*, engine_args, usage_context):
        captured["engine_args"] = engine_args
        return _DummyEngine()

    monkeypatch.setattr(
        "vllm.entrypoints.llm.LLMEngine.from_engine_args",
        fake_from_engine_args,
    )
    monkeypatch.setattr("vllm.entrypoints.llm.log_non_default_args", lambda *_: None)

    LLM(
        model="facebook/opt-125m",
        save_decode_cache=True,
        kv_transfer_config={
            "kv_connector": "LMCacheConnectorV1",
            "kv_role": "kv_both",
        },
    )

    kv_transfer_config = captured["engine_args"].kv_transfer_config
    assert isinstance(kv_transfer_config, KVTransferConfig)
    assert kv_transfer_config.kv_connector == "LMCacheConnectorV1"
    assert kv_transfer_config.kv_role == "kv_both"
    assert kv_transfer_config.save_decode_cache is True
