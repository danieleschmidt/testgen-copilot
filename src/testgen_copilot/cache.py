"""Caching infrastructure for TestGen Copilot performance optimization."""

import os
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from .logging_config import get_generator_logger


class CacheEntry:
    """Single cache entry with metadata."""

    def __init__(self, value: Any, file_path: Union[str, Path], ttl_seconds: int = 3600):
        self.value = value
        self.file_path = Path(file_path)
        self.created_at = time.time()
        self.ttl_seconds = ttl_seconds
        self.access_count = 1
        self.last_accessed = time.time()

        # Store file metadata for invalidation
        try:
            stat = os.stat(self.file_path)
            self.file_size = stat.st_size
            self.file_mtime = stat.st_mtime
        except (OSError, FileNotFoundError):
            # File might not exist, cache should be invalidated
            self.file_size = -1
            self.file_mtime = -1

    def is_valid(self) -> bool:
        """Check if cache entry is still valid."""
        # Check TTL
        if time.time() - self.created_at > self.ttl_seconds:
            return False

        # Check if file has been modified
        try:
            stat = os.stat(self.file_path)
            if stat.st_size != self.file_size or stat.st_mtime != self.file_mtime:
                return False
        except (OSError, FileNotFoundError):
            # File no longer exists, cache is invalid
            return False

        return True

    def access(self) -> Any:
        """Mark as accessed and return value."""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.value


class LRUCache:
    """Thread-safe LRU cache with file modification time invalidation."""

    def __init__(self, max_size: int = 128, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self.logger = get_generator_logger()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.invalidations = 0

    def _make_key(self, file_path: Union[str, Path], operation: str = "default") -> str:
        """Create cache key from file path and operation."""
        return f"{operation}:{str(Path(file_path).resolve())}"

    def get(self, file_path: Union[str, Path], operation: str = "default") -> Optional[Any]:
        """Get cached value if valid."""
        key = self._make_key(file_path, operation)

        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None

            entry = self._cache[key]
            if not entry.is_valid():
                self.invalidations += 1
                del self._cache[key]
                return None

            self.hits += 1
            return entry.access()

    def put(self, file_path: Union[str, Path], value: Any, operation: str = "default") -> None:
        """Store value in cache."""
        key = self._make_key(file_path, operation)

        with self._lock:
            # Create new entry
            entry = CacheEntry(value, file_path, self.ttl_seconds)
            self._cache[key] = entry

            # Enforce size limit
            self._evict_if_needed()

            self.logger.debug("Cache entry stored", {
                "operation": operation,
                "file_path": str(file_path),
                "cache_size": len(self._cache),
                "key": key
            })

    def _evict_if_needed(self) -> None:
        """Evict least recently used entries if over size limit."""
        while len(self._cache) > self.max_size:
            # Find least recently used entry
            lru_key = min(self._cache.keys(),
                         key=lambda k: self._cache[k].last_accessed)
            del self._cache[lru_key]

            self.logger.debug("Cache entry evicted", {
                "evicted_key": lru_key,
                "new_cache_size": len(self._cache)
            })

    def invalidate(self, file_path: Union[str, Path], operation: str = "default") -> bool:
        """Manually invalidate cache entry."""
        key = self._make_key(file_path, operation)

        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self.invalidations += 1
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0
            self.invalidations = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

            return {
                "hits": self.hits,
                "misses": self.misses,
                "invalidations": self.invalidations,
                "hit_rate_percent": round(hit_rate, 2),
                "cache_size": len(self._cache),
                "max_size": self.max_size
            }


# Global cache instances
_ast_cache = LRUCache(max_size=256, ttl_seconds=3600)  # 1 hour TTL for AST
_file_content_cache = LRUCache(max_size=128, ttl_seconds=1800)  # 30 min TTL for file content
_analysis_cache = LRUCache(max_size=64, ttl_seconds=3600)  # 1 hour TTL for analysis results


def cached_operation(operation_name: str, cache_instance: Optional[LRUCache] = None):
    """Decorator to cache function results based on file path parameter."""

    def decorator(func: Callable) -> Callable:
        cache = cache_instance or _analysis_cache

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract file_path from arguments (assume first arg or 'file_path' kwarg)
            file_path = None
            if args and len(args) > 0:
                # Try first argument
                potential_path = args[0]
                if isinstance(potential_path, (str, Path)):
                    file_path = potential_path

            if not file_path and 'file_path' in kwargs:
                file_path = kwargs['file_path']

            if not file_path and 'path' in kwargs:
                file_path = kwargs['path']

            if not file_path:
                # Can't cache without file path, call function directly
                return func(*args, **kwargs)

            # Try to get from cache
            cached_result = cache.get(file_path, operation_name)
            if cached_result is not None:
                return cached_result

            # Cache miss, compute result
            result = func(*args, **kwargs)

            # Store in cache
            cache.put(file_path, result, operation_name)

            return result

        return wrapper
    return decorator


def get_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all cache instances."""
    return {
        "ast_cache": _ast_cache.get_stats(),
        "file_content_cache": _file_content_cache.get_stats(),
        "analysis_cache": _analysis_cache.get_stats()
    }


def clear_all_caches() -> None:
    """Clear all caches."""
    _ast_cache.clear()
    _file_content_cache.clear()
    _analysis_cache.clear()


# Public cache instances for direct use
ast_cache = _ast_cache
file_content_cache = _file_content_cache
analysis_cache = _analysis_cache
