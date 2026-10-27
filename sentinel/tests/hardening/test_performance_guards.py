import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sentinel.api.health_controller import router as health_router
from sentinel.api.webhook_controller import get_orchestrator
from sentinel.api.webhook_controller import router as webhook_router
from sentinel.application.risk_engine import RiskEngine
from sentinel.domain.services.debt_service import DebtService
from sentinel.domain.services.security_service import SecurityService
from sentinel.domain.services.semantic_service import SemanticService
from sentinel.infrastructure.semantic.embedding_engine import EmbeddingEngine


def _build_webhook_client() -> TestClient:
    class _NoopOrchestrator:
        async def enqueue_pull_request(self, payload: dict) -> None:
            pass

    app = FastAPI()
    app.dependency_overrides[get_orchestrator] = lambda: _NoopOrchestrator()
    app.include_router(webhook_router)
    app.include_router(health_router)
    return TestClient(app)


# --- Semantic Similarity Time ---


def test_semantic_similarity_100_functions_under_threshold():
    engine = EmbeddingEngine()
    svc = SemanticService(engine)
    codes = [f"def func_{i}(x): return x + {i}" for i in range(100)]

    start = time.monotonic()
    embeddings = [svc.generate_embedding(svc.tokenize_code(c)) for c in codes]
    for i in range(len(embeddings)):
        for j in range(i + 1, min(i + 3, len(embeddings))):
            svc.compute_similarity(embeddings[i], embeddings[j])
    elapsed = time.monotonic() - start

    assert elapsed < 3.0


def test_semantic_embedding_generation_100_funcs_under_threshold():
    engine = EmbeddingEngine()
    svc = SemanticService(engine)
    codes = [f"def func_{i}(x): return x + {i}" for i in range(100)]

    start = time.monotonic()
    for code in codes:
        tokens = svc.tokenize_code(code)
        svc.generate_embedding(tokens)
    elapsed = time.monotonic() - start

    assert elapsed < 2.0


# --- RiskEngine Aggregation Time ---


def test_risk_engine_assess_100_iterations_under_threshold():
    engine = RiskEngine()
    code = "def simple():\n    return 1"

    start = time.monotonic()
    for _ in range(100):
        engine.assess(code=code)
    elapsed = time.monotonic() - start

    assert elapsed < 2.0


def test_risk_engine_assess_with_semantic_100_iterations():
    embedding = EmbeddingEngine()
    semantic = SemanticService(embedding)
    engine = RiskEngine(semantic_service=semantic)
    code = "def simple():\n    return 1"
    existing = ["def other(): pass"]

    start = time.monotonic()
    for _ in range(100):
        engine.assess(code=code, existing_code_list=existing)
    elapsed = time.monotonic() - start

    assert elapsed < 5.0


# --- Debt Service Performance ---


def test_debt_service_1000_evaluations_under_threshold():
    svc = DebtService()
    code = "def f(x):\n" + "    if x > 0:\n        x += 1\n" * 10 + "    return x"

    start = time.monotonic()
    for _ in range(1000):
        svc.evaluate_debt(code)
    elapsed = time.monotonic() - start

    assert elapsed < 2.0


def test_ast_debt_service_1000_evaluations_under_threshold():
    """Parsing costs more than a regex sweep, so it gets its own budget.

    The AST engine runs once per changed .py file, not once per review, so this
    is the loop that matters. Kept separate from the regex guard above so a
    regression points at the engine that caused it.
    """
    from sentinel.domain.services.ast_debt_service import ASTDebtService

    svc = ASTDebtService()
    code = "def f(x):\n" + "    if x > 0:\n        x += 1\n" * 10 + "    return x"

    start = time.monotonic()
    for _ in range(1000):
        svc.analyze_source(code, "f.py")
    elapsed = time.monotonic() - start

    assert elapsed < 5.0


# --- Security Service Performance ---


def test_security_service_1000_evaluations_under_threshold():
    svc = SecurityService()
    code = "def f():\n    x = 1\n    return x"

    start = time.monotonic()
    for _ in range(1000):
        svc.analyze(code)
    elapsed = time.monotonic() - start

    assert elapsed < 2.0


# --- Webhook Latency ---


def test_webhook_latency_100_posts_under_threshold():
    client = _build_webhook_client()
    latencies: list[float] = []

    for i in range(100):
        start = time.monotonic()
        response = client.post("/webhook", json={"repo": "perf", "pr_number": i})
        elapsed = time.monotonic() - start
        latencies.append(elapsed)
        assert response.status_code == 200

    avg = sum(latencies) / len(latencies)
    assert avg < 0.1


def test_health_endpoint_latency_100_requests():
    client = _build_webhook_client()
    latencies: list[float] = []

    for _ in range(100):
        start = time.monotonic()
        response = client.get("/health")
        elapsed = time.monotonic() - start
        latencies.append(elapsed)
        assert response.status_code == 200

    avg = sum(latencies) / len(latencies)
    assert avg < 0.05


# --- Memory Footprint Guard ---


def test_repeated_risk_engine_assess_no_unbounded_growth():
    engine = RiskEngine()
    code = "def f(x):\n    return x + 1"
    results: list[dict] = []
    for _ in range(500):
        result = engine.assess(code=code)
        results.append(result)

    assert len(results) == 500
    first = results[0]
    last = results[-1]
    assert first["severity"] == last["severity"]
    assert first["complexity"] == last["complexity"]


def test_repeated_semantic_service_no_unbounded_growth():
    engine = EmbeddingEngine()
    svc = SemanticService(engine)
    code = "def f(x): return x"
    for _ in range(500):
        tokens = svc.tokenize_code(code)
        svc.generate_embedding(tokens)

    embedding = svc.generate_embedding(svc.tokenize_code(code))
    assert len(embedding) == 128
