# Chatterbox MCP Server

Lightweight Model Context Protocol (MCP) server for Chatterbox TTS, designed to run on resource-constrained VPS while providing full TTS capabilities to LLM agents.

## Features

- **🪶 Lightweight**: Proxy mode uses ~100MB RAM, perfect for $5/mo VPS
- **🔌 Flexible Backends**: Proxy to remote service OR run locally
- **🎙️ Voice Management**: Upload and manage voice samples via MCP tools
- **📝 Smart Chunking**: Automatically handles long-form text
- **🌍 Multilingual**: Support for 23 languages
- **🎭 Emotion Control**: Adjust speech expressiveness
- **🔒 Voice Cloning**: Zero-shot voice cloning from samples

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│ LLM Agent   │─────▶│ MCP Server   │─────▶│ TTS Backend     │
│ (Claude,    │      │ (Lightweight)│      │ (GPU Server)    │
│  GPT, etc.) │      │ ~100MB RAM   │      │ ~8GB RAM        │
└─────────────┘      └──────────────┘      └─────────────────┘
```

### Deployment Modes

1. **Proxy Mode** (Recommended for VPS)
   - MCP server forwards requests to remote backend
   - Resource usage: ~100MB RAM, minimal CPU
   - Perfect for: Cheap VPS + separate GPU server

2. **Lazy Loading Mode**
   - Loads model on first request
   - Unloads after idle timeout
   - Resource usage: 100MB idle, ~8GB when active
   - Perfect for: Mid-tier VPS with occasional use

3. **Always-Loaded Mode**
   - Keeps model in memory
   - Fastest response time
   - Resource usage: ~8GB RAM always
   - Perfect for: Dedicated GPU server

## 🆓 Free Tier Setup (Recommended!)

Want to get started for just **$5/month**? Deploy the MCP server on a cheap VPS and use HuggingFace Spaces as a free TTS backend!

### Quick Setup (10 minutes)

1. **Deploy Chatterbox to HuggingFace Space** (free GPU backend)
2. **Deploy MCP server to VPS** ($5/month - DigitalOcean, Linode, etc.)
3. **Configure MCP to use your Space**

**Total cost: $5/month** (vs $50+ for dedicated GPU server!)

👉 **[Complete HuggingFace Space Setup Guide →](../docs/HUGGINGFACE_SPACE_SETUP.md)**

The guide includes:
- Step-by-step HF Space deployment
- Ready-to-use Gradio app template
- MCP server configuration
- Troubleshooting tips
- Cost optimization strategies

---

## Quick Start

### 1. Install Dependencies

```bash
# Minimal (for proxy mode)
pip install mcp httpx pydantic pyyaml

# Or install full package
pip install -e .
```

### 2. Configure

```bash
# Copy example config
cp mcp_config.example.yaml mcp_config.yaml

# Edit configuration (see Configuration section below)
nano mcp_config.yaml
```

### 3. Run Server

```bash
# Run directly
python -m mcp_server.server

# Or with Docker
docker build -f Dockerfile.mcp -t chatterbox-mcp .
docker run -p 8000:8000 -v ./voices:/app/voices chatterbox-mcp
```

### 4. Use from LLM

The MCP server exposes these tools to LLM agents:

```python
# Generate speech
generate_speech(
    text="Hello, this is a test",
    voice_id="abc123",
    exaggeration=0.6
)

# Upload voice sample
upload_voice(
    name="My Voice",
    file_path="/path/to/voice.wav",
    language="en"
)

# List available voices
list_voices()

# Get backend status
backend_info()
```

## Configuration

### Proxy Mode (Lightweight VPS)

```yaml
mode: "proxy"

proxy:
  backend_url: "http://gpu-server.example.com:7860"
  api_key: "${TTS_API_KEY}"
  timeout: 120
  max_retries: 3
```

**Resource Requirements:**
- RAM: 512MB - 1GB
- CPU: 1 core
- Storage: 10GB
- Cost: ~$5/month (DigitalOcean, Linode, Vultr)

### Lazy Loading Mode (Mid-tier VPS)

```yaml
mode: "lazy"

local:
  device: "cpu"  # or "cuda" if GPU available
  model_type: "english"
  lazy_unload_timeout: 600  # 10 minutes
  max_concurrent: 1
```

**Resource Requirements:**
- RAM: 8GB+
- CPU: 2-4 cores
- Storage: 20GB
- Cost: ~$40/month

### Always-Loaded Mode (Dedicated Server)

```yaml
mode: "always"

local:
  device: "cuda"  # requires GPU
  model_type: "multilingual"
  max_concurrent: 2
```

**Resource Requirements:**
- RAM: 8GB+
- GPU: CUDA-capable (NVIDIA)
- Storage: 20GB
- Cost: ~$50+/month (RunPod, Vast.ai, etc.)

## MCP Tools Reference

### generate_speech

Generate speech audio from text.

```python
generate_speech(
    text: str,                    # Required: Text to synthesize
    voice_id: str = None,         # Optional: Voice ID from list_voices
    language_id: str = None,      # Optional: Language code (e.g., 'fr', 'es')
    exaggeration: float = 0.5,    # Emotional intensity (0.25-2.0)
    cfg_weight: float = 0.5,      # Guidance weight (0.0-1.0)
    temperature: float = 0.8      # Sampling temperature
) -> audio_bytes
```

**Returns:** Base64-encoded WAV audio

**Example:**
```python
audio = generate_speech(
    text="Bonjour, comment allez-vous?",
    language_id="fr",
    exaggeration=0.7
)
```

### upload_voice

Upload a voice sample for cloning.

```python
upload_voice(
    name: str,                    # Required: Voice name
    file_path: str,               # Required: Path to audio file
    language: str = None,         # Optional: Language code
    description: str = None       # Optional: Description
) -> voice_id
```

**Supported formats:** WAV, MP3, FLAC, OGG
**Max size:** 10MB (configurable)

### list_voices

List all available voices.

```python
list_voices(
    language: str = None  # Optional: Filter by language
) -> List[Voice]
```

### get_voice_info

Get detailed information about a voice.

```python
get_voice_info(
    voice_id: str  # Required: Voice ID
) -> VoiceInfo
```

### backend_info

Get backend status and statistics.

```python
backend_info() -> BackendInfo
```

## Deployment Examples

### Example 1: HuggingFace Space + Cheap VPS ⭐ RECOMMENDED

**Setup: FREE TTS backend + $5/month VPS**

This is the most cost-effective deployment option!

**Step 1: Deploy to HuggingFace Space** (free GPU backend)

```bash
# Use the ready-to-deploy Space template
cd huggingface_space/

# Follow the guide to deploy to HF Spaces
# See: docs/HUGGINGFACE_SPACE_SETUP.md
```

Your Space URL will be: `https://YOUR_USERNAME-chatterbox-tts.hf.space`

**Step 2: Deploy MCP Server to VPS**

```yaml
# mcp_config.yaml
mode: "proxy"
proxy:
  backend_url: "https://YOUR_USERNAME-chatterbox-tts.hf.space"
  timeout: 180  # HF Spaces can be slower on CPU tier
```

```bash
# On your VPS ($5/month - DigitalOcean, Linode, Vultr)
docker run -d -p 8000:8000 \
  -v ./voices:/app/voices \
  -v ./mcp_config.yaml:/app/mcp_config.yaml \
  --restart unless-stopped \
  chatterbox-mcp
```

**Monthly cost:** ~$5 (VPS only, HF Space is FREE!)

**Performance:**
- CPU Space (free): 30-60s per generation
- GPU upgrade ($0.60/hr): 3-5s per generation

👉 **[Full HuggingFace Space Setup Guide →](../docs/HUGGINGFACE_SPACE_SETUP.md)**

### Example 2: All-in-One Docker Compose

**Setup:** Run both MCP server and TTS backend together

```bash
# Start both services
docker-compose up -d

# MCP server: http://localhost:8000
# Gradio UI: http://localhost:7860
```

**Requirements:** Machine with NVIDIA GPU

### Example 3: Serverless Backend

**Setup:** MCP server on VPS, serverless TTS backend

```yaml
# mcp_config.yaml
mode: "proxy"
proxy:
  backend_url: "https://api.modal.com/my-tts-endpoint"
  api_key: "${MODAL_API_KEY}"
```

**Monthly cost:** $5 VPS + pay-per-use serverless

## Voice Sample Best Practices

### Recording Quality

- **Duration:** 5-10 seconds recommended
- **Sample rate:** 24kHz or higher
- **Format:** WAV (uncompressed) preferred
- **Content:** Clear speech, minimal background noise
- **Language:** Match language of target speech

### Example Voice Samples

```bash
# Upload English voice
upload_voice(
    name="Professional Narrator",
    file_path="./samples/narrator.wav",
    language="en",
    description="Deep, clear voice for narration"
)

# Upload French voice
upload_voice(
    name="French Speaker",
    file_path="./samples/french.wav",
    language="fr",
    description="Native French speaker"
)
```

## Troubleshooting

### MCP Server won't start

```bash
# Check logs
docker logs chatterbox-mcp

# Verify config
python -c "from mcp_server.config import ChatterboxConfig; ChatterboxConfig.from_yaml('mcp_config.yaml')"
```

### Backend connection failed

```bash
# Test backend manually
curl -X POST http://backend-url/api/predict \
  -H "Content-Type: application/json" \
  -d '{"data": ["test", null, 0.5, 0.5, 0.8]}'
```

### Model loading slow/OOM

```bash
# Switch to lazy mode or reduce concurrency
mode: "lazy"
local:
  max_concurrent: 1
  lazy_unload_timeout: 300  # Unload after 5 minutes
```

### Voice upload fails

```bash
# Check file size
ls -lh voice.wav  # Should be < 10MB

# Check format
file voice.wav    # Should be WAV, MP3, FLAC, or OGG

# Check permissions
chmod 644 voice.wav
```

## Performance

### Benchmarks (GPU inference)

- **Short text (50 chars):** ~2-3 seconds
- **Medium text (200 chars):** ~5-8 seconds
- **Long text (1000 chars):** ~25-30 seconds
- **RTF (Real-time factor):** 0.3-0.5x (faster than real-time)

### Benchmarks (CPU inference)

- **Short text (50 chars):** ~10-15 seconds
- **Medium text (200 chars):** ~30-40 seconds
- **RTF:** 3-5x (slower than real-time)

## Development

### Running tests

```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/
```

### Adding new backends

```python
from mcp_server.backends.base import TTSBackend

class MyBackend(TTSBackend):
    async def generate(self, request):
        # Your implementation
        pass
```

## Support

- GitHub Issues: https://github.com/resemble-ai/chatterbox/issues
- Discord: https://discord.gg/rJq9cRJBJ6

## License

MIT License - see LICENSE file
