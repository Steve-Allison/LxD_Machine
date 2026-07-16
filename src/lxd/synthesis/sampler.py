"""Client-side sampling seam for synthesis.

The MCP server can either run the synthesis LLM call itself (calling
Ollama via the server's own credentials) or delegate to the client via
``ctx.sample`` (MCP sampling), letting the client's LLM answer with the
client's own model choice and token budget.

The seam is a sync callable — :data:`Sampler` — that the MCP server
constructs and hands to :func:`lxd.synthesis.answering.synthesize_answer`.
The sampler is sync because synthesis runs inside a worker thread via
:mod:`lxd.mcp.async_runtime`; the server bridges its async
``ctx.sample`` call back onto the event loop via
:func:`anyio.from_thread.run`.

When the sampler raises :class:`SamplerFailure` — e.g. the connected
client did not advertise sampling capability, or the client's own model
returned an error — the caller falls back to the server-side Ollama
path and records the reason in the answer envelope's ``warnings``.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SamplerRequest:
    """One synthesis request handed to a :data:`Sampler`.

    Attributes:
        prompt: The fully-composed synthesis prompt (preamble + graph
            context + evidence + question), same string the server-side
            Ollama path would send.
        temperature: Sampling temperature to request from the client.
        max_tokens: Maximum tokens the sampler should generate.
    """

    prompt: str
    temperature: float
    max_tokens: int


class SamplerFailure(RuntimeError):
    """Raised by a :data:`Sampler` when the client sampling call fails.

    Callers catch this and fall back to the server-side synthesis path
    with the failure message surfaced as a warning on the resulting
    :class:`lxd.synthesis.answering.AnswerEnvelope`.
    """


Sampler = Callable[[SamplerRequest], str]
"""Sync callable that runs one synthesis request against the client's LLM.

Contract:
    * Called with a :class:`SamplerRequest`; returns the plain answer text.
    * Raises :class:`SamplerFailure` when the client cannot fulfil the
      request (no sampling capability, upstream error, empty response).
    * Any other exception propagates unchanged and is treated as a
      genuine synthesis error, not a capability miss.
"""
