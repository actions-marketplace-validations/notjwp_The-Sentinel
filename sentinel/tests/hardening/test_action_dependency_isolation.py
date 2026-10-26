"""The GitHub Action must import under its own, narrower dependency set.

`requirements-action.txt` deliberately omits fastapi, uvicorn and redis: the
action runs one review and exits, so it needs no web server and no durable
queue. Nothing in the normal test environment notices when a server-only
import leaks into that path, because every one of those packages is installed
here — the failure appears for the first time in a consumer's workflow, which
is the worst place to find it. It has happened once already, when the LLM
quota added a module-level `redis` import to the worker.

Run in a subprocess with those packages blocked at the import hook, which is a
truthful simulation: the action fails the same way whether a module is absent
or unimportable.
"""

import subprocess
import sys

_BLOCKED = ("redis", "fastapi", "uvicorn")

_SCRIPT = """
import sys

BLOCKED = set(%r)


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(
                "No module named %%r (not in requirements-action.txt)" %% name
            )
        return None


sys.meta_path.insert(0, _Blocker())

import sentinel.action.runner as runner

# Exercise the real entry point, not just the import: a deferred import inside
# a function would otherwise slip through untested.
code = runner.main([], {})
print("EXIT", code)
"""


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )


def test_the_action_runs_without_the_server_dependencies():
    result = _run(_SCRIPT % (_BLOCKED,))

    assert result.returncode == 0, (
        "the action could not run under requirements-action.txt:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    # Exit 2 is "no pull request context", which is the correct answer to an
    # empty environment and proves main() ran rather than blew up on import.
    assert "EXIT 2" in result.stdout


def test_the_blocker_itself_works():
    """Guards the guard: a test that cannot fail proves nothing."""
    result = _run("import sys\n" + _SCRIPT % (("sentinel",),))

    assert result.returncode != 0
    assert "No module named" in result.stderr
