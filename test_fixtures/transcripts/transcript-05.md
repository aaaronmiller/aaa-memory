Human: Error handling strategy?

Assistant: Decision: Use structured error responses with error codes. Decision: All errors logged to centralized system.

Human: Retry logic?

Assistant: Decision: Exponential backoff with jitter, max 3 retries.