from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from forge_krea2_depth.preprocessor_cache import (
    ControlMapCache,
    build_control_map_cache_key,
    prepare_control_source,
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


def test_prepared_source_is_a_stable_snapshot_of_mutable_input():
    source = np.full((4, 5, 3), 17, dtype=np.uint8)

    snapshot, fingerprint = prepare_control_source(source)
    source[:] = 99

    assert np.all(snapshot == 17)
    assert source_fingerprint(snapshot) == fingerprint
    assert source_fingerprint(source) != fingerprint


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


def test_structured_sequence_maps_are_stored_losslessly_in_compressed_form():
    cache = ControlMapCache(max_entries=4, max_bytes=4096)
    source = np.zeros((8, 8, 3), dtype=np.uint8)
    result = np.zeros((128, 128, 3), dtype=np.uint8)
    result[20:108, 62:66] = (255, 80, 20)

    first, first_hit = cache.get_or_compute(key(source), lambda: result.copy())
    second, second_hit = cache.get_or_compute(
        key(source), lambda: np.full_like(result, 99)
    )

    assert first_hit is False
    assert second_hit is True
    assert np.array_equal(first, result)
    assert np.array_equal(second, result)
    assert cache.info()["bytes"] < result.nbytes // 10


def test_unreadable_cached_payload_falls_back_to_fresh_preprocessing():
    cache = ControlMapCache(max_entries=4, max_bytes=4096)
    cache_key = key(np.zeros((8, 8, 3), dtype=np.uint8))
    initial = np.zeros((64, 64, 3), dtype=np.uint8)
    cache.get_or_compute(cache_key, lambda: initial)
    cache._entries[cache_key].payload = b"not a png"
    calls = []

    restored, hit = cache.get_or_compute(
        cache_key,
        lambda: calls.append(True) or np.full((64, 64, 3), 77, dtype=np.uint8),
    )

    assert hit is False
    assert len(calls) == 1
    assert np.all(restored == 77)


def test_compressed_long_sequence_survives_a_second_full_scan():
    cache = ControlMapCache()
    calls = []
    keys = []
    expected = []
    for index in range(300):
        source = np.zeros((2, 2, 3), dtype=np.uint8)
        source[0, 0, :2] = (index // 256, index % 256)
        keys.append(key(source))
        result = np.zeros((64, 64, 3), dtype=np.uint8)
        result[:, index % 64] = (index % 256, 255, 40)
        expected.append(result)
        cache.get_or_compute(
            keys[-1], lambda index=index, result=result: calls.append(index) or result
        )

    for index, (cache_key, result) in enumerate(zip(keys, expected)):
        restored, hit = cache.get_or_compute(
            cache_key, lambda index=index: calls.append(index) or np.full_like(result, 99)
        )
        assert hit is True
        assert np.array_equal(restored, result)

    assert calls == list(range(300))
    assert cache.info()["entries"] == 300


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


def test_reconfiguring_limits_evicts_immediately_and_zero_bypasses_cache():
    cache = ControlMapCache(max_entries=3, max_bytes=1024)
    calls = []
    for value in (1, 2, 3):
        cache.get_or_compute(
            key(np.full((1, 1, 3), value)),
            lambda value=value: np.full((2, 2, 3), value, dtype=np.uint8),
        )

    cache.configure(max_entries=1, max_bytes=1024)
    assert cache.info()["entries"] == 1
    assert cache.info()["evictions"] == 2

    cache.configure(max_entries=0, max_bytes=1024)
    assert cache.info()["entries"] == 0
    bypass_key = key(np.zeros((1, 1, 3)))
    for _ in range(2):
        cache.get_or_compute(
            bypass_key,
            lambda: calls.append(True) or np.zeros((2, 2, 3), dtype=np.uint8),
        )
    assert len(calls) == 2
    assert cache.info()["entries"] == 0


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
