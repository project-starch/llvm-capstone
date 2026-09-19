"""Common trace inspection; format adapters preserve allocator-specific semantics."""

from .model import Event, TraceError
from .reader import TraceReader, inspect_trace, record_trace

__all__ = ["Event", "TraceError", "TraceReader", "inspect_trace", "record_trace"]
