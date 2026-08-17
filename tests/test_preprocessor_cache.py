from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from forge_krea2_depth.preprocessor_cache import (
    ControlMapCache,
    build_control_map_cache_key,
    source_fingerprint,
)


def key(value, resolution=768):
    return build_control_map_cache_key(
        value, "Depth", "depth_anything_v2", resolution, False
    )


def test_fingerprint_uses_pixels_not_path_and_detects_in_place_changes(tmp_path):
    first = tmp_path / "first.png"
    copy = tmp_path / "copy.png"
    pixels = np.full((4, 5, 3), 17, dtype=np.uint8)
    Image.fromarray(pixels).save(first)
    Image.fromarray(pixels).save(copy)

    initial = source_fingerprint(first)
    assert source_fingerprint(copy) == initial

    Image.fromarray(np.full_like(pixels, 18)).save(first)
    assert source_fingerprint(first) != initial


def test_cache_reuses_results_without_exposing_its_stored_array():
    cache = ControlMapCache(max_entries=2, max_bytes=1024)
    calls = []

    def compute():
        calls.append(True)
        return np.full((2, 3, 3), 7, dtype=np.uint8)

    first, first_hit = cache.get_or_compute(key(np.zeros((2, 2, 3))), compute)
    first[:] = 99
    second, second_hit = cache.get_or_compute(key(np.zeros((2, 2, 3))), compute)

    assert first_hit is False
    assert second_hit is True
    assert len(calls) == 1
    assert np.all(second == 7)


def test_lru_and_byte_limits_clean_old_entries_without_flushing_on_oversize():
    cache = ControlMapCache(max_entries=2, max_bytes=30)
    compute = lambda value: lambda: np.full((2, 2, 3), value, dtype=np.uint8)
    one, two, three = (key(np.full((1, 1, 3), value)) for value in (1, 2, 3))
    cache.get_or_compute(one, compute(1))
    cache.get_or_compute(two, compute(2))
    cache.get_or_compute(one, compute(1))
    cache.get_or_compute(three, compute(3))

    assert cache.info()["entries"] == 2
    assert cache.info()["evictions"] == 1
    _result, hit = cache.get_or_compute(two, compute(2))
    assert hit is False

    resident_before = cache.info()["entries"]
    oversized = key(np.full((1, 1, 3), 4))
    cache.get_or_compute(
        oversized, lambda: np.zeros((20, 20, 3), dtype=np.uint8)
    )
    assert cache.info()["entries"] == resident_before


def test_concurrent_same_key_runs_the_preprocessor_once():
    cache = ControlMapCache(max_entries=2, max_bytes=1024)
    started = threading.Event()
    release = threading.Event()
    calls = []

    def compute():
        calls.append(True)
        started.set()
        assert release.wait(timeout=2)
        return np.ones((2, 2, 3), dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(cache.get_or_compute, key(np.zeros((2, 2, 3))), compute)
        assert started.wait(timeout=2)
        second = pool.submit(cache.get_or_compute, key(np.zeros((2, 2, 3))), compute)
        release.set()
        first_result = first.result(timeout=2)
        second_result = second.result(timeout=2)

    assert len(calls) == 1
    assert sorted((first_result[1], second_result[1])) == [False, True]


def test_failed_or_cleared_inflight_work_never_poison_the_cache():
    cache = ControlMapCache(max_entries=2, max_bytes=1024)
    cache_key = key(np.zeros((2, 2, 3)))

    try:
        cache.get_or_compute(cache_key, lambda: (_ for _ in ()).throw(ValueError("bad")))
    except ValueError:
        pass
    assert cache.info()["entries"] == 0
    assert cache.info()["pending"] == 0

    started = threading.Event()
    release = threading.Event()

    def compute():
        started.set()
        assert release.wait(timeout=2)
        return np.ones((2, 2, 3), dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(cache.get_or_compute, cache_key, compute)
        assert started.wait(timeout=2)
        cache.clear()
        release.set()
        future.result(timeout=2)

    assert cache.info()["entries"] == 0
