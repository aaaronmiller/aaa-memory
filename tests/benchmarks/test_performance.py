"""Performance benchmarks for retrieval latency."""
import time, pytest

def test_hot_search_latency():
    from aaa_memory.retrieval.hot import search
    start = time.time()
    results = search("test query", limit=5)
    elapsed = time.time() - start
    assert elapsed < 5.0, f"Hot search took {elapsed:.2f}s (expected <5s)"
    print(f"Hot search: {elapsed:.3f}s")

def test_cold_search_latency():
    from aaa_memory.retrieval.cold import search_archive
    start = time.time()
    results = search_archive("test", limit=5)
    elapsed = time.time() - start
    print(f"Cold search: {elapsed:.3f}s")
