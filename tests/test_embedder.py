"""Tests for ctx_semantic.embedder (T4).

The model load is module-scoped (~1.5 s warm cache; models/ is the pinned
622 MB artifact from task-1-scaffold.md — never deleted by tests). GPU usage
inside Embedder is verified via nvidia-smi PID lookup, not provider names
(the task-1 silent-fallback trap).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ctx_semantic.embedder import DEFAULT_CACHE_DIR, DIM, Embedder

CACHE_DIR = Path(DEFAULT_CACHE_DIR)


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    e = Embedder(cache_dir=CACHE_DIR)
    e.embed_query("warmup")  # lazy load once for the whole module
    return e


def test_dim_is_768(embedder: Embedder):
    v = embedder.embed_query("维度检查")
    assert DIM == 768
    assert v.shape == (768,)


def test_device_report_line(embedder: Embedder):
    # "cuda" is only set after the nvidia-smi PID check passed inside
    # _ensure_loaded; a silent CPU fallback would have logged + flipped this.
    assert embedder.device == "cuda", f"device reported: {embedder.device}"


def test_bilingual_cosine_sanity(embedder: Embedder):
    # jina-v2-base-zh is zh/en bilingual: the zh query must match its English
    # paraphrase far better than an unrelated concept. Strict inequality with
    # a 0.2 margin — measured gap is ~0.68, so this is robust to model
    # determinism noise, not a knife-edge threshold.
    q = embedder.embed_query("模型路由")
    pos = embedder.embed_query("model routing 分流")
    neg = embedder.embed_query("晚餐食谱")
    cos_pos = float(np.dot(q, pos))
    cos_neg = float(np.dot(q, neg))
    assert cos_pos > cos_neg, f"{cos_pos=} !> {cos_neg=}"
    assert cos_pos - cos_neg > 0.2, f"gap {cos_pos - cos_neg:.4f} too small"


def test_embed_batch_contract(embedder: Embedder):
    texts = ["标题\n\n内容第一段", "model routing", "晚餐食谱 今日菜单"]
    mat = embedder.embed_batch(texts)
    assert mat.shape == (3, DIM)
    assert mat.dtype == np.float32
    assert np.allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-5)


def test_embed_batch_empty():
    mat = Embedder().embed_batch([])
    assert mat.shape == (0, DIM)
    assert mat.dtype == np.float32


def test_query_unit_norm_float32(embedder: Embedder):
    v = embedder.embed_query("单位范数检查")
    assert v.dtype == np.float32
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5
