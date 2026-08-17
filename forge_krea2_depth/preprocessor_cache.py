"""Persistent, content-addressed cache for Krea control preprocessors."""

from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from forge_krea2_depth.images import normalize_image


CACHE_SCHEMA_VERSION = 1
DEFAULT_MAX_ENTRIES = 128
DEFAULT_MAX_BYTES = 256 * 1024 * 1024


def _array_digest(value: np.ndarray) -> bytes:
    array = np.ascontiguousarray(value)
    digest = hashlib.blake2b(digest_size=20)
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.digest()


def source_fingerprint(source: Any) -> bytes:
    """Fingerprint decoded pixels, so paths and in-place file changes are safe."""

    normalized_source = os.fspath(source) if isinstance(source, os.PathLike) else source
    try:
        return _array_digest(normalize_image(normalized_source))
    except ValueError:
        # Tests and third-party API wrappers can use opaque source handles which
        # their patched preprocessors understand. Real invalid images still fail
        # in the preprocessor, and failures are never inserted into the cache.
        if isinstance(source, (str, os.PathLike)):
            digest = hashlib.blake2b(digest_size=20)
            digest.update(type(source).__name__.encode("ascii"))
            digest.update(os.fspath(source).encode("utf-8", errors="surrogatepass"))
            return digest.digest()
        raise


@dataclass(frozen=True)
class ControlMapCacheKey:
    schema_version: int
    source_digest: bytes
    mode: str
    preprocessor: str
    resolution: int
    invert: bool


def build_control_map_cache_key(
    source: Any,
    mode: str,
    preprocessor: str,
    resolution: int,
    invert: bool,
) -> ControlMapCacheKey:
    return ControlMapCacheKey(
        schema_version=CACHE_SCHEMA_VERSION,
        source_digest=source_fingerprint(source),
        mode=str(mode),
        preprocessor=str(preprocessor),
        resolution=int(resolution),
        invert=bool(invert),
    )


@dataclass
class _CacheEntry:
    result: np.ndarray
    size_bytes: int


class ControlMapCache:
    """Memory-bounded thread-safe LRU with one in-flight job per cache key."""

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self._lock = threading.RLock()
        self._entries: OrderedDict[ControlMapCacheKey, _CacheEntry] = OrderedDict()
        self._pending: dict[ControlMapCacheKey, threading.Event] = {}
        self._max_entries = max(0, int(max_entries))
        self._max_bytes = max(0, int(max_bytes))
        self._current_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._generation = 0

    @property
    def enabled(self) -> bool:
        return self._max_entries > 0 and self._max_bytes > 0

    def configure(self, max_entries: int, max_bytes: int) -> None:
        with self._lock:
            self._max_entries = max(0, int(max_entries))
            self._max_bytes = max(0, int(max_bytes))
            self._evict_to_limits()

    def clear(self, reset_stats: bool = False) -> None:
        with self._lock:
            # Results already being computed may still return to their caller,
            # but this generation token prevents them repopulating a cleared cache.
            self._generation += 1
            self._entries.clear()
            self._current_bytes = 0
            if reset_stats:
                self._hits = 0
                self._misses = 0
                self._evictions = 0

    def info(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self._current_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "pending": len(self._pending),
                "max_entries": self._max_entries,
                "max_bytes": self._max_bytes,
            }

    def get_or_compute(
        self,
        key: ControlMapCacheKey,
        compute: Callable[[], np.ndarray],
    ) -> tuple[np.ndarray, bool]:
        while True:
            with self._lock:
                if not self.enabled:
                    bypass_cache = True
                    break
                entry = self._entries.get(key)
                if entry is not None:
                    self._entries.move_to_end(key)
                    self._hits += 1
                    return entry.result.copy(), True
                pending = self._pending.get(key)
                if pending is None:
                    pending = threading.Event()
                    self._pending[key] = pending
                    owner_generation = self._generation
                    bypass_cache = False
                    break
            pending.wait()

        if bypass_cache:
            return compute(), False

        try:
            result = compute()
        except BaseException:
            with self._lock:
                self._pending.pop(key, None)
                pending.set()
            raise

        with self._lock:
            self._misses += 1
            try:
                if (
                    self.enabled
                    and owner_generation == self._generation
                    and isinstance(result, np.ndarray)
                ):
                    cached = np.ascontiguousarray(result).copy()
                    size_bytes = cached.nbytes
                    # A single oversized map is returned normally without
                    # evicting useful resident sequence entries it cannot replace.
                    if size_bytes <= self._max_bytes:
                        previous = self._entries.pop(key, None)
                        if previous is not None:
                            self._current_bytes -= previous.size_bytes
                        self._entries[key] = _CacheEntry(cached, size_bytes)
                        self._current_bytes += size_bytes
                        self._evict_to_limits()
            except Exception:
                # Caching is only an optimisation; a valid inference must survive
                # allocation or accounting failures.
                pass
            finally:
                self._pending.pop(key, None)
                pending.set()
        return result, False

    def _evict_to_limits(self) -> None:
        while self._entries and (
            not self.enabled
            or len(self._entries) > self._max_entries
            or self._current_bytes > self._max_bytes
        ):
            _, entry = self._entries.popitem(last=False)
            self._current_bytes -= entry.size_bytes
            self._evictions += 1
