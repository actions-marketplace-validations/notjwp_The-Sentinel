import os
import time
from typing import Any

try:  # pragma: no cover - absence is handled in _build_client
    import openai
except ImportError:  # pragma: no cover
    openai = None

from sentinel.infrastructure.llm.base import LLMProvider
from sentinel.monitoring.logger import get_logger


class NIMProvider(LLMProvider):
    BASE_URL = "https://integrate.api.nvidia.com/v1"
    MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
    RETRY_DELAYS = (0.3, 0.8)

    # The default model reasons before it answers, and the reasoning is billed
    # against the same completion budget as the answer. At 700 the budget was
    # spent entirely on reasoning: the response came back with content=None and
    # every review silently fell through to static text. A real two-finding
    # audit needs ~1300 completion tokens, so this leaves headroom rather than
    # sitting just above the observed figure.
    MAX_TOKENS = 2048

    # Measured: a two-finding audit against the default model takes ~18s wall
    # clock, because reasoning tokens are generated before the first word of
    # the answer. The old 5.0 was tuned for a non-reasoning model and would
    # have cut every call off mid-thought. Enrichment runs off the request
    # path (a worker thread, or the Action's own job), so waiting is cheap;
    # silently degrading to static text is not.
    DEFAULT_TIMEOUT = 45.0

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        base_url: str = BASE_URL,
        model: str = MODEL,
        client: Any = None,
    ) -> None:
        self.logger = get_logger(__name__)
        self.api_key = api_key if api_key is not None else os.getenv("NVIDIA_API_KEY", "").strip()
        self.timeout = max(0.0, timeout)
        self.base_url = base_url
        self.model = model
        self._client = client if client is not None else self._build_client()

    @staticmethod
    def _sleep(seconds: float) -> None:
        """Isolated so tests can neutralize retry backoff without real waiting."""
        try:
            time.sleep(seconds)
        except Exception:  # pragma: no cover - defensive
            return

    def _build_client(self) -> Any | None:
        if not self.api_key:
            return None
        if openai is None:
            self.logger.warning("openai package is not installed; LLM calls disabled.")
            return None

        openai_cls = getattr(openai, "OpenAI", None)
        if openai_cls is None:
            return None

        try:
            return openai_cls(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=0,
            )
        except Exception:
            return None

    @staticmethod
    def _sanitize(text: str | None) -> str | None:
        if text is None:
            return None
        cleaned = text.replace("\u0000", "").strip()
        return cleaned or None

    @staticmethod
    def _extract_content(response: Any) -> str | None:
        choices = getattr(response, "choices", None)
        if not choices:
            return None

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None and isinstance(first_choice, dict):
            message = first_choice.get("message")

        content = None
        if message is not None:
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")

        return NIMProvider._sanitize(content)

    @staticmethod
    def _finish_reason(response: Any) -> str:
        """Why generation stopped, for the empty-content log line.

        `length` here means the model never reached its answer — on a reasoning
        model the whole budget went on thinking. That is a configuration
        problem with a one-line fix, and it is indistinguishable from a genuine
        refusal unless the reason is logged, so it is.
        """
        choices = getattr(response, "choices", None)
        if not choices:
            return "unknown"
        first = choices[0]
        reason = getattr(first, "finish_reason", None)
        if reason is None and isinstance(first, dict):
            reason = first.get("finish_reason")
        return str(reason) if reason else "unknown"

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return True
        error_code = getattr(exc, "code", None)
        if error_code == 429:
            return True
        message = str(exc).lower()
        return "429" in message or "rate limit" in message

    def _chat(self, system_prompt: str, user_prompt: str) -> str | None:
        if not self.api_key or self._client is None:
            return None

        for attempt, delay in enumerate((0.0,) + self.RETRY_DELAYS):
            if attempt:
                self._sleep(delay)
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=self.MAX_TOKENS,
                    stream=False,
                )
            except Exception as exc:
                if self._is_rate_limit_error(exc) and attempt < len(self.RETRY_DELAYS):
                    self.logger.warning("NIM rate limit encountered; retrying.")
                    continue
                self.logger.warning("NIM request failed.")
                return None

            content = self._extract_content(response)
            if content:
                return content
            self.logger.warning(
                "NIM response missing content (finish_reason=%s, max_tokens=%s). "
                "A reasoning model that stops on 'length' spent the budget thinking.",
                self._finish_reason(response),
                self.MAX_TOKENS,
            )
            return None

        return None

    def generate_text(self, prompt: str) -> str | None:
        """One-shot free-form completion; reuses _chat's retry/sanitize handling."""
        if not isinstance(prompt, str) or not prompt.strip():
            return None
        return self._chat("", prompt)

    def generate_pr_audit(self, code: str, findings_summary: str) -> str | None:
        prompt = (
            "You are a senior secure code reviewer. Review the pull request and the "
            "detected issues below.\n\n"
            f"Code:\n{code}\n\n"
            f"Detected Issues:\n{findings_summary}\n\n"
            "For EACH detected issue, output one block in EXACTLY this format:\n\n"
            "Issue: <short title>\n"
            "Explanation: <one or two concise sentences>\n"
            "Fix: <corrected code only>\n\n"
            "Each issue lists the rule that fired, the line, and the exact matched "
            "code. Explain and fix THAT code. The bracketed rule name is a detector "
            "id, not a variable in the program — never treat it as one.\n\n"
            "Formatting rules — follow them exactly:\n"
            "- Use the labels 'Issue:', 'Explanation:', and 'Fix:' verbatim, each at the "
            "start of its own line.\n"
            "- Do NOT use markdown: no **bold**, no headings (#), no bullet points, no "
            "numbering before the labels.\n"
            "- Put the fix code on the line(s) after 'Fix:' with no code fences (no ```).\n"
            "- Emit one block per issue, in the same order as listed above, and output "
            "nothing else."
        )
        return self._chat("", prompt)
