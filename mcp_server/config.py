"""
Configuration management for Chatterbox MCP Server.
"""

from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field
import yaml
import os


class ProxyConfig(BaseModel):
    """Configuration for proxy mode (forwards to remote backend)."""
    backend_url: str = Field(..., description="URL of remote TTS backend")
    api_key: Optional[str] = Field(None, description="API key for authentication")
    timeout: int = Field(120, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum retry attempts")


class LocalConfig(BaseModel):
    """Configuration for local inference mode."""
    device: Literal["cpu", "cuda", "mps"] = Field("cpu", description="Compute device")
    model_type: Literal["english", "multilingual"] = Field("english", description="Model variant")
    lazy_unload_timeout: int = Field(600, description="Seconds before unloading idle model")
    max_concurrent: int = Field(1, description="Maximum concurrent generations")


class VoiceConfig(BaseModel):
    """Voice sample storage configuration."""
    storage: Literal["local", "s3"] = Field("local", description="Storage backend")
    local_path: str = Field("./voices", description="Local storage path")
    max_size_mb: int = Field(10, description="Maximum voice sample size in MB")


class QueueConfig(BaseModel):
    """Request queue configuration."""
    enabled: bool = Field(False, description="Enable request queueing")
    backend: Literal["memory", "redis"] = Field("memory", description="Queue backend")
    redis_url: Optional[str] = Field(None, description="Redis connection URL")
    max_queue_size: int = Field(10, description="Maximum queue length")


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = Field("0.0.0.0", description="Bind host")
    port: int = Field(8000, description="Bind port")
    workers: int = Field(1, description="Number of workers")
    log_level: Literal["debug", "info", "warning", "error"] = Field("info", description="Logging level")


class ChatterboxConfig(BaseModel):
    """Complete Chatterbox MCP Server configuration."""
    mode: Literal["proxy", "lazy", "always"] = Field("proxy", description="Deployment mode")
    proxy: ProxyConfig = Field(default_factory=lambda: ProxyConfig(backend_url="http://localhost:7860"))
    local: LocalConfig = Field(default_factory=LocalConfig)
    voices: VoiceConfig = Field(default_factory=VoiceConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "ChatterboxConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Expand environment variables
        data = cls._expand_env_vars(data)

        return cls(**data)

    @classmethod
    def _expand_env_vars(cls, obj):
        """Recursively expand environment variables in config."""
        if isinstance(obj, dict):
            return {k: cls._expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [cls._expand_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            return os.getenv(var_name, obj)
        return obj

    @classmethod
    def default(cls) -> "ChatterboxConfig":
        """Create default configuration (proxy mode)."""
        return cls()

    def save(self, path: Path):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


# Default configuration
DEFAULT_CONFIG = ChatterboxConfig.default()
