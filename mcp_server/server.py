"""
Chatterbox MCP Server - Lightweight TTS for LLM agents.

Exposes MCP tools for:
- generate_speech: Generate audio from text
- upload_voice: Add voice samples
- list_voices: List available voices
- get_voice_info: Get voice metadata
- backend_info: Get backend status
"""

import asyncio
import logging
import base64
from pathlib import Path
from typing import Optional, Any

from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource

from .config import ChatterboxConfig
from .backends import RemoteBackend, LocalBackend, TTSRequest
from .voice_manager import VoiceManager
from smart_chunking import SmartChunker
from long_form_tts import LongFormTTS

logger = logging.getLogger(__name__)


class ChatterboxMCPServer:
    """
    Lightweight MCP server for Chatterbox TTS.

    Designed to run on resource-constrained VPS while providing
    full TTS capabilities to LLM agents.
    """

    def __init__(self, config: ChatterboxConfig):
        """
        Args:
            config: Server configuration
        """
        self.config = config
        self.server = Server("chatterbox-tts")

        # Initialize components
        self.voice_manager = VoiceManager(
            db_path=config.voices.local_path + "/voices.db",
            storage_dir=config.voices.local_path
        )

        # Initialize backend based on mode
        if config.mode == "proxy":
            self.backend = RemoteBackend(
                backend_url=config.proxy.backend_url,
                api_key=config.proxy.api_key,
                timeout=config.proxy.timeout,
                max_retries=config.proxy.max_retries,
            )
        else:  # lazy or always
            lazy_mode = (config.mode == "lazy")
            self.backend = LocalBackend(
                device=config.local.device,
                model_type=config.local.model_type,
                lazy_mode=lazy_mode,
                unload_timeout=config.local.lazy_unload_timeout,
                max_concurrent=config.local.max_concurrent,
            )

        # Initialize utilities
        self.chunker = SmartChunker(max_chars=250)

        logger.info(f"ChatterboxMCPServer initialized in {config.mode} mode")

        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register MCP tools."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="generate_speech",
                    description=(
                        "Generate speech audio from text using TTS. "
                        "Supports voice cloning, emotion control, and multiple languages. "
                        "For long text (>250 chars), automatically chunks at natural boundaries."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to synthesize (can be arbitrarily long)",
                            },
                            "voice_id": {
                                "type": "string",
                                "description": "Voice ID from list_voices (optional, uses default if not specified)",
                            },
                            "language_id": {
                                "type": "string",
                                "description": "Language code (e.g., 'en', 'fr', 'es') for multilingual model",
                            },
                            "exaggeration": {
                                "type": "number",
                                "description": "Emotional intensity (0.25-2.0, default 0.5=neutral)",
                                "default": 0.5,
                            },
                            "cfg_weight": {
                                "type": "number",
                                "description": "Guidance weight for pacing (0.0-1.0, default 0.5)",
                                "default": 0.5,
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Sampling temperature (0.05-5.0, default 0.8)",
                                "default": 0.8,
                            },
                        },
                        "required": ["text"],
                    },
                ),
                Tool(
                    name="upload_voice",
                    description=(
                        "Upload a voice sample for voice cloning. "
                        "Accepts audio files (WAV, MP3, FLAC, OGG) up to 10MB. "
                        "Returns voice_id for use in generate_speech."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Human-readable name for this voice",
                            },
                            "file_path": {
                                "type": "string",
                                "description": "Path to audio file",
                            },
                            "language": {
                                "type": "string",
                                "description": "Language code (e.g., 'en', 'fr')",
                            },
                            "description": {
                                "type": "string",
                                "description": "Optional description of the voice",
                            },
                        },
                        "required": ["name", "file_path"],
                    },
                ),
                Tool(
                    name="list_voices",
                    description="List all available voice samples",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "language": {
                                "type": "string",
                                "description": "Filter by language (optional)",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_voice_info",
                    description="Get detailed information about a voice sample",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "voice_id": {
                                "type": "string",
                                "description": "Voice ID",
                            },
                        },
                        "required": ["voice_id"],
                    },
                ),
                Tool(
                    name="backend_info",
                    description="Get backend status and information",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent | EmbeddedResource]:
            """Execute tool."""

            if name == "generate_speech":
                return await self._generate_speech(**arguments)

            elif name == "upload_voice":
                return await self._upload_voice(**arguments)

            elif name == "list_voices":
                return await self._list_voices(**arguments)

            elif name == "get_voice_info":
                return await self._get_voice_info(**arguments)

            elif name == "backend_info":
                return await self._backend_info()

            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _generate_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language_id: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
    ) -> list[TextContent | EmbeddedResource]:
        """Generate speech from text."""

        # Get voice sample path
        voice_path = None
        if voice_id:
            voice = self.voice_manager.get_voice(voice_id)
            if not voice:
                return [TextContent(
                    type="text",
                    text=f"Error: Voice ID '{voice_id}' not found. Use list_voices to see available voices."
                )]
            voice_path = voice.file_path

        # Build request
        request = TTSRequest(
            text=text,
            voice_path=voice_path,
            language_id=language_id,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )

        try:
            # Generate audio
            logger.info(f"Generating speech: {len(text)} chars, voice={voice_id}")
            response = await self.backend.generate(request)

            # Encode audio as base64
            audio_b64 = base64.b64encode(response.audio_data).decode('utf-8')

            # Return as embedded resource
            return [
                TextContent(
                    type="text",
                    text=f"Generated {response.duration_seconds:.1f}s of audio ({len(text)} chars)"
                ),
                EmbeddedResource(
                    type="resource",
                    resource={
                        "uri": f"data:audio/wav;base64,{audio_b64}",
                        "mimeType": "audio/wav",
                        "blob": audio_b64,
                    }
                )
            ]

        except Exception as e:
            logger.exception("Generation failed")
            return [TextContent(
                type="text",
                text=f"Error: Generation failed - {str(e)}"
            )]

    async def _upload_voice(
        self,
        name: str,
        file_path: str,
        language: Optional[str] = None,
        description: Optional[str] = None,
    ) -> list[TextContent]:
        """Upload voice sample."""
        try:
            voice = self.voice_manager.add_voice(
                name=name,
                file_path=file_path,
                language=language,
                description=description,
                max_size_mb=self.config.voices.max_size_mb,
            )

            return [TextContent(
                type="text",
                text=(
                    f"Voice '{name}' uploaded successfully!\n"
                    f"Voice ID: {voice.voice_id}\n"
                    f"Duration: {voice.duration_seconds:.1f}s\n"
                    f"Sample rate: {voice.sample_rate}Hz\n"
                    f"Size: {voice.file_size_bytes / 1024:.1f}KB\n\n"
                    f"Use this voice_id in generate_speech calls."
                )
            )]

        except Exception as e:
            logger.exception("Upload failed")
            return [TextContent(
                type="text",
                text=f"Error: Upload failed - {str(e)}"
            )]

    async def _list_voices(
        self,
        language: Optional[str] = None,
    ) -> list[TextContent]:
        """List voice samples."""
        voices = self.voice_manager.list_voices(language=language)

        if not voices:
            return [TextContent(
                type="text",
                text="No voices found. Use upload_voice to add voice samples."
            )]

        # Format as table
        lines = ["Available Voices:", ""]
        lines.append(f"{'ID':<18} {'Name':<20} {'Lang':<6} {'Duration':<8}")
        lines.append("-" * 60)

        for voice in voices:
            duration = f"{voice.duration_seconds:.1f}s" if voice.duration_seconds else "N/A"
            lang = voice.language or "N/A"
            lines.append(f"{voice.voice_id:<18} {voice.name:<20} {lang:<6} {duration:<8}")

        lines.append("")
        lines.append(f"Total: {len(voices)} voices")

        return [TextContent(type="text", text="\n".join(lines))]

    async def _get_voice_info(
        self,
        voice_id: str,
    ) -> list[TextContent]:
        """Get voice information."""
        voice = self.voice_manager.get_voice(voice_id)

        if not voice:
            return [TextContent(
                type="text",
                text=f"Voice ID '{voice_id}' not found."
            )]

        lines = [
            f"Voice: {voice.name}",
            f"ID: {voice.voice_id}",
            f"Language: {voice.language or 'N/A'}",
            f"Duration: {voice.duration_seconds:.1f}s" if voice.duration_seconds else "Duration: N/A",
            f"Sample rate: {voice.sample_rate}Hz" if voice.sample_rate else "Sample rate: N/A",
            f"File size: {voice.file_size_bytes / 1024:.1f}KB" if voice.file_size_bytes else "File size: N/A",
            f"File path: {voice.file_path}",
            f"Description: {voice.description or 'N/A'}",
            f"Created: {voice.created_at}",
        ]

        return [TextContent(type="text", text="\n".join(lines))]

    async def _backend_info(self) -> list[TextContent]:
        """Get backend information."""
        info = await self.backend.get_info()

        lines = [
            "Backend Information:",
            "",
            f"Mode: {self.config.mode}",
            f"Type: {info.get('backend_type', 'unknown')}",
            f"Healthy: {info.get('healthy', False)}",
        ]

        if info.get('backend_type') == 'remote':
            lines.extend([
                f"Backend URL: {info.get('backend_url', 'N/A')}",
                f"Timeout: {info.get('timeout', 'N/A')}s",
            ])
        elif info.get('backend_type') == 'local':
            lines.extend([
                f"Device: {info.get('device', 'N/A')}",
                f"Model: {info.get('model_type', 'N/A')}",
                f"Loaded: {info.get('model_loaded', False)}",
                f"Idle time: {info.get('idle_time', 0):.0f}s" if 'idle_time' in info else "",
            ])

        # Voice stats
        stats = self.voice_manager.get_stats()
        lines.extend([
            "",
            "Voice Statistics:",
            f"Total voices: {stats['total_voices']}",
            f"Total size: {stats['total_size_mb']:.1f}MB",
            f"By language: {stats['by_language']}",
        ])

        return [TextContent(type="text", text="\n".join(lines))]

    async def cleanup(self):
        """Cleanup resources."""
        await self.backend.cleanup()

    def run(self):
        """Run the MCP server."""
        logger.info("Starting Chatterbox MCP Server...")
        self.server.run()


async def main():
    """Main entry point."""
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    config_path = Path("mcp_config.yaml")
    if config_path.exists():
        config = ChatterboxConfig.from_yaml(config_path)
        logger.info(f"Loaded config from {config_path}")
    else:
        config = ChatterboxConfig.default()
        logger.warning("Using default configuration (proxy mode)")

    # Create and run server
    server = ChatterboxMCPServer(config)

    try:
        server.run()
    finally:
        await server.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
