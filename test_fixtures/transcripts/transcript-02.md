Human: We need a caching layer.

Assistant: Redis would work well here. Decision: Use Redis with LRU eviction.

Human: Cache invalidation strategy?

Assistant: Decision: Time-based TTL with manual invalidation on writes.