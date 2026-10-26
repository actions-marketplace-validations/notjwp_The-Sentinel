"""Running a review as a GitHub Action.

A third delivery mechanism alongside ``api/`` (HTTP webhook) and ``workers/``
(queue), sharing the same engines. The difference is where the work happens and
whose credentials pay for it: here the consumer's own runner executes the
review under their own ``GITHUB_TOKEN`` and their own LLM key, so anyone can
use the tool without anything being hosted for them.
"""
