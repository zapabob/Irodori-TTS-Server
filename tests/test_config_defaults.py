from __future__ import annotations

from pathlib import Path

from irodori_openai_tts.config import Settings


def test_perf_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.default_decode_mode == "batch"
    assert settings.enable_cpu_fallback_on_oom is True
    assert settings.ref_latent_cache_size >= 1


def test_default_model_precision_is_fp32() -> None:
    """fp32 must stay the untouched default; bf16 needs native device support."""
    assert Settings(_env_file=None).model_precision == "fp32"


def test_env_example_keeps_fp32_default(tmp_path) -> None:
    """README tells users to `cp .env.example .env`, which overrides the code default."""
    example = Path(__file__).resolve().parents[1] / ".env.example"
    env_file = tmp_path / ".env"
    env_file.write_bytes(example.read_bytes())

    assert Settings(_env_file=str(env_file)).model_precision == "fp32"
