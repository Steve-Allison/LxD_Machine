"""Shared HTTP and network-client infrastructure for LxD.

This package owns long-lived HTTP clients that should be reused across
callers. Per-call client instantiation imposes TCP/TLS handshake overhead
on every request and disables HTTP/2 multiplexing; the factories here
return module-level singletons keyed on the relevant connection axis
(base_url/timeout/limits) so that every module in the process shares the
same pool.
"""

from __future__ import annotations
