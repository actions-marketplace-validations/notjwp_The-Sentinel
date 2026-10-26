from sentinel.infrastructure.llm.nim_provider import NIMProvider


class _Message:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str | None) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str | None) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, response: _Response | None = None, should_raise: bool = False) -> None:
        self.response = response or _Response("ok")
        self.should_raise = should_raise
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.should_raise:
            raise RuntimeError("request failed")
        return self.response


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class _Client:
    def __init__(self, completions: _Completions) -> None:
        self.chat = _Chat(completions)


def test_generate_pr_audit_returns_clean_text_from_nim_response():
    completions = _Completions(response=_Response("  Issue:\nExplanation: e\nFix: f  "))
    provider = NIMProvider(api_key="k", client=_Client(completions), timeout=10.0)
    assert provider.generate_pr_audit("code", "summary") == "Issue:\nExplanation: e\nFix: f"

    call = completions.calls[0]
    assert call["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert call["stream"] is False


def test_generate_pr_audit_returns_none_on_api_error():
    completions = _Completions(should_raise=True)
    provider = NIMProvider(api_key="k", client=_Client(completions))
    assert provider.generate_pr_audit("code", "summary") is None


def test_returns_none_when_missing_api_key():
    provider = NIMProvider(api_key="", client=_Client(_Completions()))
    assert provider.generate_pr_audit("code", "summary") is None


def test_returns_none_on_invalid_payload_shape():
    provider = NIMProvider(api_key="k", client=_Client(_Completions(response=_Response(None))))
    assert provider.generate_pr_audit("code", "summary") is None


# --- Reasoning models: the budget pays for thinking before it pays for answers ---


class _FinishedChoice(_Choice):
    def __init__(self, content: str | None, finish_reason: str) -> None:
        super().__init__(content)
        self.finish_reason = finish_reason


class _FinishedResponse:
    def __init__(self, content: str | None, finish_reason: str) -> None:
        self.choices = [_FinishedChoice(content, finish_reason)]


def test_the_token_budget_leaves_room_for_an_answer_after_the_reasoning():
    """Pinned because 700 silently disabled enrichment for every review.

    The default model emits reasoning tokens before the first word of its
    answer, billed against the same completion budget. A measured two-finding
    audit needs ~1300; at 700 the response came back with content=None and the
    report fell through to static text with nothing in the log to say why.
    """
    completions = _Completions()
    NIMProvider(api_key="k", client=_Client(completions)).generate_pr_audit("code", "summary")

    assert completions.calls[0]["max_tokens"] == NIMProvider.MAX_TOKENS
    assert NIMProvider.MAX_TOKENS >= 1500


def test_the_timeout_outlasts_a_reasoning_model():
    """5.0 was sized for a model that answers immediately; this one takes ~18s."""
    assert NIMProvider.DEFAULT_TIMEOUT >= 30.0


def test_empty_content_logs_the_finish_reason_that_explains_it(caplog):
    """`length` is a config error with a one-line fix. Unlogged, it reads as a refusal."""
    completions = _Completions(response=_FinishedResponse(None, "length"))
    provider = NIMProvider(api_key="k", client=_Client(completions))

    with caplog.at_level("WARNING"):
        assert provider.generate_pr_audit("code", "summary") is None

    logged = caplog.text
    assert "finish_reason=length" in logged
    assert str(NIMProvider.MAX_TOKENS) in logged


def test_finish_reason_is_reported_as_unknown_rather_than_crashing():
    """The provider must never raise — a response without the field is still handled."""
    assert NIMProvider._finish_reason(_Response(None)) == "unknown"
    assert NIMProvider._finish_reason(object()) == "unknown"
