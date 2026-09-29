from __future__ import annotations

import asyncio
from pathlib import Path

import torch

from irodori_openai_tts import app
from irodori_openai_tts.ref_cache import RefLatentCache
from irodori_tts.inference_runtime import SamplingRequest


class _DummyCodec:
    sample_rate = 24000

    def encode_waveform(self, wav, sample_rate, normalize_db, ensure_max):  # noqa: ANN001
        _ = (sample_rate, normalize_db, ensure_max)
        return torch.zeros((1, 12, 32), dtype=torch.float32)


class _DummyRuntime:
    def __init__(self) -> None:
        self.codec = _DummyCodec()


def test_is_cuda_oom_runtime_error_detection() -> None:
    assert app._is_cuda_oom_runtime_error(RuntimeError("CUDA out of memory")) is True
    assert app._is_cuda_oom_runtime_error(RuntimeError("OutOfMemoryError")) is True
    assert app._is_cuda_oom_runtime_error(RuntimeError("other runtime error")) is False


def test_runtime_load_oom_uses_cpu_fallback(monkeypatch) -> None:
    calls: list[str] = []
    cpu_runtime = object()

    def fail_gpu_load():
        calls.append("gpu")
        raise RuntimeError("CUDA out of memory")

    def load_cpu_fallback():
        calls.append("cpu")
        return cpu_runtime

    monkeypatch.setattr(app.runtime_manager, "get", fail_gpu_load)
    monkeypatch.setattr(app.runtime_manager, "get_cpu_fallback", load_cpu_fallback)

    result = asyncio.run(app._get_runtime_with_cpu_fallback())

    assert result is cpu_runtime
    assert calls == ["gpu", "cpu"]


def test_ref_latent_cache_reuses_same_reference(tmp_path, monkeypatch) -> None:
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"dummy")
    runtime = _DummyRuntime()
    cache = RefLatentCache(max_entries=2)

    def fake_load(path: str):  # noqa: ANN001
        assert Path(path) == voice_path
        return torch.zeros((1, 24000), dtype=torch.float32), 24000

    monkeypatch.setattr("irodori_openai_tts.ref_cache.torchaudio.load", fake_load)

    req = SamplingRequest(text="test", ref_wav=str(voice_path), no_ref=False)
    first = cache.resolve_or_create(runtime, req)
    second = cache.resolve_or_create(runtime, req)

    assert first is not None
    assert second == first
    assert Path(first).is_file()
