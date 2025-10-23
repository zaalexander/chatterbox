"""
TTS backend implementations.
"""

from .base import TTSBackend, TTSRequest, TTSResponse, BackendError
from .remote import RemoteBackend
from .local import LocalBackend

__all__ = [
    "TTSBackend",
    "TTSRequest",
    "TTSResponse",
    "BackendError",
    "RemoteBackend",
    "LocalBackend",
]
