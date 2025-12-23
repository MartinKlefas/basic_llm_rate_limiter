import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

try:
    import tiktoken
except ImportError:
    tiktoken = None


@dataclass
class LimiterState:
    req_expiries: Deque[float]
    tok_expiries: Deque[Tuple[float, int]]
    tok_sum: int


class SlidingWindowRateLimiter:
    """
    Sliding-window rate limiter for:
      - requests per window (RPM over 60s, etc.)
      - tokens per window (TPM over 60s, etc.)

    Process-local: safe for asyncio tasks *within one process*.
    Not shared across processes/machines unless you back state with Redis/etc.
    """

    def __init__(
        self,
        *,
        max_requests: int,
        max_tokens: int,
        window_seconds: float = 60.0,
        model_for_tokenizer: str = "gpt-4o-mini",
        fallback_chars_per_token: float = 4.0,
    ):
        self.max_requests = int(max_requests)
        self.max_tokens = int(max_tokens)
        self.window = float(window_seconds)

        self._lock = asyncio.Lock()
        self._state = LimiterState(
            req_expiries=deque(),
            tok_expiries=deque(),
            tok_sum=0,
        )

        self._fallback_cpt = float(fallback_chars_per_token)
        self._enc = None
        if tiktoken is not None:
            try:
                # tiktoken model mappings change; best-effort
                self._enc = tiktoken.encoding_for_model(model_for_tokenizer)
            except Exception:
                self._enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is not None:
            return len(self._enc.encode(text))
        # crude fallback if tiktoken not installed
        return max(1, int(len(text) / self._fallback_cpt))

    def _evict_expired(self, now: float) -> None:
        # requests
        while self._state.req_expiries and self._state.req_expiries[0] <= now:
            self._state.req_expiries.popleft()

        # tokens
        while self._state.tok_expiries and self._state.tok_expiries[0][0] <= now:
            exp, tok = self._state.tok_expiries.popleft()
            self._state.tok_sum -= tok

    def _next_wait_time(self, now: float, tokens_needed: int) -> float:
        """
        If we can't consume now, compute how long to wait until we can.
        """
        # if RPM exceeded, wait until earliest request expires
        wait_rpm = 0.0
        if len(self._state.req_expiries) >= self.max_requests:
            wait_rpm = max(0.0, self._state.req_expiries[0] - now)

        # if TPM exceeded, wait until enough tokens expire
        wait_tpm = 0.0
        if self._state.tok_sum + tokens_needed > self.max_tokens:
            # simulate removing token entries until we'd be under limit
            needed_to_free = (self._state.tok_sum + tokens_needed) - self.max_tokens
            freed = 0
            for exp, tok in self._state.tok_expiries:
                freed += tok
                if freed >= needed_to_free:
                    wait_tpm = max(0.0, exp - now)
                    break
            else:
                # should be rare; but if it happens, wait for the earliest token expiry
                if self._state.tok_expiries:
                    wait_tpm = max(0.0, self._state.tok_expiries[0][0] - now)

        return max(wait_rpm, wait_tpm)

    async def acquire(self, *, prompt: str, extra_tokens: int = 0) -> int:
        """
        Wait until we're allowed to submit a request consuming prompt tokens.
        Returns the token count reserved.

        extra_tokens: optionally reserve anticipated completion tokens too.
        """
        tokens_needed = self.count_tokens(prompt) + int(extra_tokens)
        if tokens_needed > self.max_tokens:
            raise ValueError(
                f"Single request needs {tokens_needed} tokens, exceeds window max_tokens={self.max_tokens}."
            )

        while True:
            async with self._lock:
                now = time.monotonic()
                self._evict_expired(now)

                can_rpm = len(self._state.req_expiries) < self.max_requests
                can_tpm = (self._state.tok_sum + tokens_needed) <= self.max_tokens

                if can_rpm and can_tpm:
                    expiry = now + self.window
                    self._state.req_expiries.append(expiry)
                    self._state.tok_expiries.append((expiry, tokens_needed))
                    self._state.tok_sum += tokens_needed
                    return tokens_needed

                sleep_for = self._next_wait_time(now, tokens_needed)

            # IMPORTANT: sleep outside the lock
            await asyncio.sleep(max(0.01, sleep_for))

    async def __aenter__(self):
        raise RuntimeError("Use `async with limiter.reserve(prompt=...)` instead.")

    def reserve(self, *, prompt: str, extra_tokens: int = 0):
        """
        Convenience context manager:
            async with limiter.reserve(prompt=prompt):
                ... call API ...
        """
        limiter = self

        class _Reservation:
            async def __aenter__(self_nonlocal):
                self_nonlocal.tokens = await limiter.acquire(prompt=prompt, extra_tokens=extra_tokens)
                return self_nonlocal.tokens

            async def __aexit__(self_nonlocal, exc_type, exc, tb):
                return False

        return _Reservation()
