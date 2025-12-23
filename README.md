# Basic LLM Rate Limiter

The aim here is to have a tool that enforces wait times between llm api calls based on Requests Per Minute or Tokens Per Minute limits. This should allow us to wait only when we need to and maximise throughput without getting any 429 errors.
