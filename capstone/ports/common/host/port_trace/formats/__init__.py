"""Explicit registry: adding a port does not change any existing wire format."""

from .ffmpeg import Ffmpeg
from .ggml import Ggml
from .postgres import Postgres
from .pymalloc import Pymalloc
from .wmem import Wmem

FORMATS = tuple(cls() for cls in (Ffmpeg, Postgres, Pymalloc, Ggml, Wmem))
