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
