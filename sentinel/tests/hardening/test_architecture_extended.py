import ast
import importlib
import sys
from pathlib import Path

SENTINEL_ROOT = Path(__file__).resolve().parents[2]
DOMAIN_DIR = SENTINEL_ROOT / "domain"
INFRASTRUCTURE_DIR = SENTINEL_ROOT / "infrastructure"


def _python_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*.py"))


def _parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _extract_imports(tree: ast.AST) -> list[str]:
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        if isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


# --- Domain Must Not Import Forbidden Libraries ---


DOMAIN_FORBIDDEN = [
    "fastapi",
    "sklearn",
    "pinecone",
    "requests",
    "subprocess",
    "sentinel.api",
    "sentinel.workers",
    "sentinel.application",
    "sentinel.infrastructure",
    "logging",
]


def test_domain_has_no_forbidden_imports():
    for path in _python_files(DOMAIN_DIR):
        tree = _parse(path)
        imports = _extract_imports(tree)
        for imp in imports:
            for forbidden in DOMAIN_FORBIDDEN:
                assert not imp.startswith(forbidden), (
                    f"{path.name} imports '{imp}' which is forbidden "
                    f"(matches '{forbidden}')"
                )


def test_domain_has_no_sklearn_imports():
    for path in _python_files(DOMAIN_DIR):
        tree = _parse(path)
        imports = _extract_imports(tree)
        for imp in imports:
            assert not imp.startswith("sklearn"), (
                f"{path.name} imports '{imp}'"
            )


def test_domain_has_no_pinecone_imports():
    for path in _python_files(DOMAIN_DIR):
        tree = _parse(path)
        imports = _extract_imports(tree)
        for imp in imports:
            assert not imp.startswith("pinecone"), (
                f"{path.name} imports '{imp}'"
            )


def test_domain_has_no_requests_imports():
    for path in _python_files(DOMAIN_DIR):
        tree = _parse(path)
        imports = _extract_imports(tree)
        for imp in imports:
            assert imp != "requests" and not imp.startswith("requests."), (
                f"{path.name} imports '{imp}'"
            )


def test_domain_has_no_subprocess_imports():
    for path in _python_files(DOMAIN_DIR):
        tree = _parse(path)
        imports = _extract_imports(tree)
        for imp in imports:
            assert imp != "subprocess" and not imp.startswith("subprocess."), (
                f"{path.name} imports '{imp}'"
            )


# --- Infrastructure Must Not Import Application Layer ---


INFRA_FORBIDDEN = [
    "sentinel.application",
    "sentinel.api",
    "sentinel.workers",
]


def test_infrastructure_has_no_application_imports():
    for path in _python_files(INFRASTRUCTURE_DIR):
        tree = _parse(path)
        imports = _extract_imports(tree)
        for imp in imports:
            for forbidden in INFRA_FORBIDDEN:
                assert not imp.startswith(forbidden), (
                    f"{path.name} imports '{imp}' which is forbidden "
                    f"(matches '{forbidden}')"
                )


# --- No Circular Imports ---


def test_no_circular_imports_in_sentinel_modules():
    modules_to_check = [
        "sentinel.domain.services.debt_service",
        "sentinel.domain.services.security_service",
        "sentinel.domain.services.semantic_service",
        "sentinel.domain.entities.finding",
        "sentinel.domain.entities.pull_request",
        "sentinel.domain.value_objects.severity_level",
        "sentinel.application.risk_engine",
        "sentinel.application.audit_orchestrator",
        "sentinel.infrastructure.semantic.embedding_engine",
        "sentinel.infrastructure.redis.redis_job_queue",
        "sentinel.infrastructure.redis.redis_delivery_dedup",
        "sentinel.infrastructure.redis.redis_llm_quota",
        "sentinel.api.webhook_controller",
        "sentinel.api.health_controller",
        "sentinel.workers.job_queue",
        "sentinel.workers.background_worker",
        "sentinel.action.context",
        "sentinel.action.runner",
        # Deliberately not sentinel.action.__main__: importing it runs main().
    ]

    for module_name in modules_to_check:
        if module_name in sys.modules:
            del sys.modules[module_name]

    for module_name in modules_to_check:
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            raise AssertionError(
                f"Circular import detected when importing {module_name}: {exc}"
            ) from exc


# --- Domain Services Must Be Synchronous ---


def test_domain_services_have_no_async_functions():
    services_dir = DOMAIN_DIR / "services"
    for path in services_dir.rglob("*.py"):
        tree = _parse(path)
        for node in ast.walk(tree):
            assert not isinstance(node, ast.AsyncFunctionDef), (
                f"{path.name} contains async function '{node.name}'"
            )


# --- Domain Must Not Use Print ---


def test_domain_has_no_print_calls():
    for path in _python_files(DOMAIN_DIR):
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "print", (
                    f"{path.name} contains print() call"
                )


# --- Infrastructure Must Only Import Allowed Dependencies ---


def test_infrastructure_imports_only_allowed():
    allowed_prefixes = [
        "sklearn",
        "numpy",
        "redis",
        "sentinel.domain",
        "sentinel.infrastructure",
        "sentinel.monitoring",
        "__future__",
        "typing",
        "re",
        "dataclasses",
        "enum",
        "collections",
        "abc",
        "math",
        "hashlib",
        "json",
        "base64",
        "os",
        "pathlib",
        "asyncio",
        "uuid",
        "time",
        # The GitHub adapter's actual transport and auth. These were always used,
        # but via `__import__` inside method bodies, which this AST scan cannot
        # see — so the guard passed while the dependencies were undeclared. M9
        # moved them to real imports; listing them here is the honest state.
        "datetime",
        "urllib",
        "jwt",
        "openai",
        # Corpus blobs are fetched concurrently: 20 independent round trips
        # were 7.8s of a 45s review, against ~0.0s for every analysis engine
        # combined. Threads only overlap I/O here — no shared mutable state
        # crosses the pool, and results are reassembled by index.
        "concurrent",
    ]

    for path in _python_files(INFRASTRUCTURE_DIR):
        tree = _parse(path)
        imports = _extract_imports(tree)
        for imp in imports:
            is_allowed = any(
                imp == prefix or imp.startswith(prefix + ".")
                for prefix in allowed_prefixes
            )
            assert is_allowed, (
                f"{path.name} imports '{imp}' which is not in the allowed list"
            )
