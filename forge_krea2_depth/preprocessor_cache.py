"""Persistent, content-addressed cache for Krea control preprocessors."""

from __future__ import annotations

import hashlib
import io
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from PIL import Image

from forge_krea2_depth.images import normalize_image


CACHE_SCHEMA_VERSION = 1
DEFAULT_MAX_ENTRIES = 512
DEFAULT_MAX_BYTES = 256 * 1024 * 1024


def _array_digest(value: np.ndarray) -> bytes:
    array = np.ascontiguousarray(value)
    digest = hashlib.blake2b(digest_size=20)
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.digest()


def prepare_control_source(source: Any) -> tuple[Any, bytes]:
    """Return one stable decoded snapshot and its content fingerprint."""

    normalized_source = os.fspath(source) if isinstance(source, os.PathLike) else source
    try:
        snapshot = normalize_image(normalized_source)
        return snapshot, _array_digest(snapshot)
    except ValueError:
        # Tests and third-party API wrappers can use opaque source handles which
        # their patched preprocessors understand. Real invalid images still fail
        # in the preprocessor, and failures are never inserted into the cache.
        if isinstance(source, (str, os.PathLike)):
            digest = hashlib.blake2b(digest_size=20)
            digest.update(type(source).__name__.encode("ascii"))
            digest.update(os.fspath(source).encode("utf-8", errors="surrogatepass"))
            return normalized_source, digest.digest()
        raise


def source_fingerprint(source: Any) -> bytes:
    """Fingerprint decoded pixels, so paths and in-place file changes are safe."""

    _snapshot, fingerprint = prepare_control_source(source)
    return fingerprint


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
    *,
    source_digest: bytes | None = None,
) -> ControlMapCacheKey:
    return ControlMapCacheKey(
        schema_version=CACHE_SCHEMA_VERSION,
        source_digest=(
            source_fingerprint(source) if source_digest is None else source_digest
        ),
        mode=str(mode),
        preprocessor=str(preprocessor),
        resolution=int(resolution),
        invert=bool(invert),
    )


@dataclass
class _CacheEntry:
    payload: np.ndarray | bytes
    compressed: bool
    size_bytes: int

    def restore(self) -> np.ndarray:
        if not self.compressed:
            return self.payload.copy()
        with Image.open(io.BytesIO(self.payload)) as image:
            return np.ascontiguousarray(np.asarray(image).copy())


def _pack_result(result: np.ndarray) -> _CacheEntry:
    cached = np.ascontiguousarray(result).copy()
    if cached.dtype == np.uint8 and (
        cached.ndim == 2 or (cached.ndim == 3 and cached.shape[2] in (3, 4))
    ):
        try:
            output = io.BytesIO()
            Image.fromarray(cached).save(output, format="PNG", compress_level=1)
            encoded = output.getvalue()
            if len(encoded) < cached.nbytes:
                return _CacheEntry(encoded, True, len(encoded))
        except Exception:
            # Unusual NumPy layouts or unsupported PIL modes simply use raw RAM.
            pass
    return _CacheEntry(cached, False, cached.nbytes)


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

    @property
    def revision(self) -> int:
        """Change whenever an explicit clear invalidates derived UI previews."""

        with self._lock:
            return self._generation

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
                    cached_entry = entry
                else:
                    cached_entry = None
                if cached_entry is not None:
                    pending = None
                else:
                    pending = self._pending.get(key)
                if pending is None:
                    if cached_entry is None:
                        pending = threading.Event()
                        self._pending[key] = pending
                        owner_generation = self._generation
                        bypass_cache = False
                        break
            if cached_entry is not None:
                try:
                    return cached_entry.restore(), True
                except Exception:
                    # A decode/allocation failure must degrade to recomputation,
                    # never turn a cache optimisation into a generation failure.
                    with self._lock:
                        if self._entries.get(key) is cached_entry:
                            self._entries.pop(key)
                            self._current_bytes -= cached_entry.size_bytes
                    continue
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

        packed = None
        if isinstance(result, np.ndarray):
            try:
                # Compression happens outside the cache lock so unrelated images
                # can still hit or start preprocessing concurrently.
                packed = _pack_result(result)
            except Exception:
                pass
        with self._lock:
            self._misses += 1
            try:
                if (
                    self.enabled
                    and owner_generation == self._generation
                    and packed is not None
                ):
                    # An incompressible oversized map is returned normally without
                    # evicting useful resident sequence entries it cannot replace.
                    if packed.size_bytes <= self._max_bytes:
                        previous = self._entries.pop(key, None)
                        if previous is not None:
                            self._current_bytes -= previous.size_bytes
                        self._entries[key] = packed
                        self._current_bytes += packed.size_bytes
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
