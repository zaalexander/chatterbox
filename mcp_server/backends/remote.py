"""
Remote backend - forwards requests to external TTS service.

This is the lightest backend option, suitable for resource-constrained VPS.
Memory usage: ~50-100MB
"""

import httpx
import io
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .base import TTSBackend, TTSRequest, TTSResponse, BackendError

logger = logging.getLogger(__name__)


class RemoteBackend(TTSBackend):
    """
    Forwards TTS requests to remote Gradio or FastAPI endpoint.

    This backend is a thin proxy that:
    1. Receives TTS request from MCP
    2. Forwards to remote backend via HTTP
    3. Returns audio response

    Resource usage: Minimal (~50MB RAM, no GPU needed)
    """

    def __init__(
        self,
        backend_url: str,
        api_key: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3
    ):
        """
        Args:
            backend_url: URL of remote TTS backend (e.g., http://gpu-server:8000/tts)
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.backend_url = backend_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

        # Create HTTP client with retries
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
        )

        logger.info(f"RemoteBackend initialized: {backend_url}")

    async def generate(self, request: TTSRequest) -> TTSResponse:
        """
        Forward generation request to remote backend.

        Supports two API types:
        1. Gradio API (predict endpoint)
        2. FastAPI (custom /generate endpoint)
        """
        try:
            # Try FastAPI endpoint first
            response = await self._call_fastapi(request)
            if response:
                return response

            # Fall back to Gradio endpoint
            response = await self._call_gradio(request)
            if response:
                return response

            raise BackendError("Failed to get response from remote backend")

        except httpx.TimeoutException:
            raise BackendError(f"Request timeout after {self.timeout}s")
        except httpx.RequestError as e:
            raise BackendError(f"Network error: {e}")
        except Exception as e:
            logger.exception("Unexpected error in remote backend")
            raise BackendError(f"Backend error: {e}")

    async def _call_fastapi(self, request: TTSRequest) -> Optional[TTSResponse]:
        """Call FastAPI-style /generate endpoint."""
        url = f"{self.backend_url}/generate"

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Prepare multipart form data
        files = {}
        data = {
            "text": request.text,
            "exaggeration": request.exaggeration,
            "cfg_weight": request.cfg_weight,
            "temperature": request.temperature,
        }

        if request.language_id:
            data["language_id"] = request.language_id

        if request.voice_path and Path(request.voice_path).exists():
            files["voice_file"] = open(request.voice_path, "rb")

        try:
            response = await self.client.post(
                url,
                data=data,
                files=files if files else None,
                headers=headers
            )

            if response.status_code == 404:
                # FastAPI endpoint not found, return None to try Gradio
                return None

            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("content-type", "")
            if "audio" in content_type or "application/octet-stream" in content_type:
                return TTSResponse(
                    audio_data=response.content,
                    sample_rate=24000,
                    metadata={"backend": "fastapi"}
                )

            return None

        except httpx.HTTPStatusError:
            return None
        finally:
            # Close any opened files
            for f in files.values():
                if hasattr(f, 'close'):
                    f.close()

    async def _call_gradio(self, request: TTSRequest) -> Optional[TTSResponse]:
        """
        Call Gradio API endpoint.

        Gradio endpoints use /api/predict format.
        """
        url = f"{self.backend_url}/api/predict"

        # Build Gradio API payload
        # Format: {"data": [text, voice_file, exaggeration, cfg_weight, ...]}
        payload = {
            "data": [
                request.text,
                request.voice_path or None,  # Can be path or None for default
                request.exaggeration,
                request.cfg_weight,
                request.temperature,
            ]
        }

        try:
            response = await self.client.post(url, json=payload)

            if response.status_code == 404:
                return None

            response.raise_for_status()
            result = response.json()

            # Gradio returns audio file path or base64
            # Extract audio data
            if "data" in result and len(result["data"]) > 0:
                audio_output = result["data"][0]

                # If it's a file path, download it
                if isinstance(audio_output, str) and audio_output.startswith("http"):
                    audio_response = await self.client.get(audio_output)
                    audio_response.raise_for_status()
                    audio_data = audio_response.content
                elif isinstance(audio_output, dict) and "data" in audio_output:
                    # Base64 encoded audio
                    import base64
                    audio_data = base64.b64decode(audio_output["data"])
                else:
                    raise BackendError("Unexpected Gradio response format")

                return TTSResponse(
                    audio_data=audio_data,
                    sample_rate=24000,
                    metadata={"backend": "gradio"}
                )

            return None

        except httpx.HTTPStatusError:
            return None

    async def health_check(self) -> bool:
        """Check if remote backend is reachable."""
        try:
            # Try to reach the backend
            response = await self.client.get(
                f"{self.backend_url}/",
                timeout=5.0
            )
            return response.status_code < 500
        except:
            return False

    async def get_info(self) -> Dict[str, Any]:
        """Get backend information."""
        is_healthy = await self.health_check()

        return {
            "backend_type": "remote",
            "backend_url": self.backend_url,
            "healthy": is_healthy,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

    async def cleanup(self):
        """Close HTTP client."""
        await self.client.aclose()
        logger.info("RemoteBackend cleaned up")
