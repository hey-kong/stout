from __future__ import annotations

import asyncio
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from openai import AsyncOpenAI
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from minisgl.benchmark.client import generate_prompt


@dataclass(frozen=True)
class RunResult:
    ttft: float
    prompt: str


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.skip(f"{name} is not set")
    return value


async def _wait_server_ready(port: int, timeout_s: float = 180.0) -> None:
    start = time.time()
    async with AsyncOpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="") as client:
        while time.time() - start < timeout_s:
            try:
                await client.models.list()
                return
            except Exception:
                await asyncio.sleep(1.0)
    raise TimeoutError(f"Server did not become ready within {timeout_s}s")


async def _measure_ttft(port: int, model: str, prompt: str) -> float:
    async with AsyncOpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="") as client:
        response = await client.chat.completions.create(
            model=model,
            stream=True,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0.0,
            extra_body={"ignore_eos": True, "top_k": 1},
        )
        start = time.perf_counter()
        async for _ in response:
            return time.perf_counter() - start
    raise RuntimeError("No token is returned by the server")


async def _run_case(
    *,
    model_path: str,
    port: int,
    use_layerwise: bool,
    input_len: int,
    evict_rounds: int,
) -> RunResult:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    random.seed(42)
    main_prompt = generate_prompt(tokenizer, input_len)
    evict_prompts = [generate_prompt(tokenizer, input_len) for _ in range(evict_rounds)]

    cmd = [
        sys.executable,
        "-m",
        "minisgl.server.launch",
        "--model-path",
        model_path,
        "--cache-type",
        "hiradix",
        "--device-mem-layout",
        "page_first",
        "--host-mem-layout",
        "page_first",
        "--hicache-ratio",
        "4",
        "--page-size",
        "16",
        "--num-pages",
        "320",
        "--max-running-requests",
        "4",
        "--cuda-graph-max-bs",
        "1",
        "--port",
        str(port),
    ]
    if not use_layerwise:
        cmd.append("--disable-layerwise")

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    try:
        await _wait_server_ready(port)
        async with AsyncOpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="") as client:
            model = (await client.models.list()).data[0].id

        # 1) prefill once: cache to HBM + DRAM
        _ = await _measure_ttft(port, model, main_prompt)

        # 2) evict HBM cache by repeatedly inserting long unique prompts
        for prompt in evict_prompts:
            _ = await _measure_ttft(port, model, prompt)

        # 3) hit the same prompt again; expected to trigger DRAM->HBM transfer path
        ttft = await _measure_ttft(port, model, main_prompt)
        return RunResult(ttft=ttft, prompt=main_prompt)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        proc.wait(timeout=30)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for hicache perf test")
def test_hicache_page_first_ttft_compare_layerwise_modes():
    """
    Perf regression test for page_first + use_layerwise=False mode.

    Scenario:
    1) prefill one 4096-token prompt;
    2) evict HBM KV cache with other long prompts;
    3) prefill the same 4096-token prompt again to trigger host->HBM transfer;
    4) compare TTFT between use_layerwise=True and use_layerwise=False.
    """
    if os.getenv("RUN_HICACHE_TTFT_TEST") != "1":
        pytest.skip("Set RUN_HICACHE_TTFT_TEST=1 to run this perf test")

    model_path = _require_env("MINISGL_TEST_MODEL")
    input_len = 4096
    evict_rounds = 4
    layerwise = asyncio.run(
        _run_case(
            model_path=model_path,
            port=1919,
            use_layerwise=True,
            input_len=input_len,
            evict_rounds=evict_rounds,
        )
    )
    no_layerwise = asyncio.run(
        _run_case(
            model_path=model_path,
            port=1929,
            use_layerwise=False,
            input_len=input_len,
            evict_rounds=evict_rounds,
        )
    )
    print(
        "[hicache-ttft] page_first:",
        f"use_layerwise=True TTFT={layerwise.ttft:.6f}s,",
        f"use_layerwise=False TTFT={no_layerwise.ttft:.6f}s",
    )
    assert layerwise.ttft > 0.0
    assert no_layerwise.ttft > 0.0
