"""
Abstract backend interface for TTS generation.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class TTSRequest:
    """Request parameters for TTS generation."""
    text: str
    voice_id: Optional[str] = None
    voice_path: Optional[str] = None  # Direct path to voice sample
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    language_id: Optional[str] = None  # For multilingual

    # Advanced parameters
    repetition_penalty: float = 1.2
    min_p: float = 0.05
    top_p: float = 1.0
    seed: Optional[int] = None


@dataclass
class TTSResponse:
    """Response from TTS generation."""
    audio_data: bytes  # WAV format
    sample_rate: int = 24000
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TTSBackend(ABC):
    """
    Abstract interface for TTS backends.

    Implementations:
    - RemoteBackend: Forwards to remote Gradio/FastAPI endpoint
    - LocalBackend: Uses local Chatterbox model (lazy or always loaded)
    """

    @abstractmethod
    async def generate(self, request: TTSRequest) -> TTSResponse:
        """
        Generate speech audio from text.

        Args:
            request: TTS request parameters

        Returns:
            TTSResponse with audio data and metadata

        Raises:
            BackendError: If generation fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if backend is healthy and ready.

        Returns:
            True if healthy, False otherwise
        """
        pass

    @abstractmethod
    async def get_info(self) -> Dict[str, Any]:
        """
        Get backend information (model type, capabilities, etc.).

        Returns:
            Dictionary with backend metadata
        """
        pass

    async def cleanup(self):
        """
        Cleanup resources (optional).

        Called when shutting down server or switching backends.
        """
        pass


class BackendError(Exception):
    """Raised when backend operations fail."""
    pass
