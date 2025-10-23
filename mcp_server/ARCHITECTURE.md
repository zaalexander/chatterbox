# Chatterbox MCP Server - Lightweight Architecture

## Design Goals

1. **Resource Efficiency**: Run on VPS with limited RAM/CPU
2. **Flexible Deployment**: Support multiple backend modes
3. **Simple Client**: Minimal dependencies for MCP server itself
4. **Scalable**: Handle multiple concurrent requests efficiently

## Deployment Modes

### Mode 1: Proxy Mode (Recommended for VPS)
**Use case**: Lightweight VPS forwards requests to GPU server

```
LLM Agent → MCP Server (VPS) → Remote TTS Backend (GPU server)
            [Lightweight]         [Heavy compute]
```

**Resource usage**: ~100MB RAM, minimal CPU
- MCP server is just a thin API wrapper
- No model loading
- Manages queue, voice samples, chunking
- Forwards to remote Gradio/FastAPI endpoint

### Mode 2: Lazy Loading Mode
**Use case**: VPS with 8GB+ RAM, occasional use

```
LLM Agent → MCP Server → Local TTS Model (loaded on-demand)
```

**Resource usage**: 100MB idle, ~8GB when active
- Load model on first request
- Keep in memory for configurable timeout (e.g., 10 minutes)
- Unload if idle to free memory
- Good for sporadic usage

### Mode 3: Always-Loaded Mode
**Use case**: Dedicated GPU server, frequent use

```
LLM Agent → MCP Server → Local TTS Model (always loaded)
```

**Resource usage**: ~8GB RAM always
- Model stays in memory
- Fastest response time
- Only for dedicated servers

## Architecture Components

### 1. MCP Server Core (Lightweight)
```python
# Minimal dependencies:
- mcp (MCP SDK)
- httpx (HTTP client for proxy mode)
- pydantic (data validation)
- redis (optional, for queue/cache)
```

**Size**: ~50MB total
**Responsibilities**:
- Expose MCP tools
- Handle authentication
- Route requests to backend
- Manage voice sample metadata

### 2. Backend Abstraction Layer
```python
class TTSBackend(ABC):
    @abstractmethod
    async def generate(self, text: str, voice_id: str, **params) -> bytes:
        pass

class RemoteBackend(TTSBackend):
    """Forwards to remote Gradio/FastAPI endpoint"""

class LocalBackend(TTSBackend):
    """Uses local Chatterbox model (lazy or always loaded)"""
```

### 3. Voice Sample Manager
- Store voice sample metadata (name, path, language)
- Upload to remote backend or store locally
- Lightweight SQLite database (~1MB)

### 4. Request Queue (Optional)
- Redis-based queue for handling concurrent requests
- Can run Redis locally or use managed service
- Fallback to in-memory queue if Redis unavailable

## Recommended VPS Setup

### Minimal VPS (Proxy Mode)
```
CPU: 1 core
RAM: 512MB - 1GB
Storage: 10GB
Cost: ~$5/month

Components:
- MCP server (Python)
- Nginx (reverse proxy)
- SQLite (voice metadata)
- systemd (process manager)

Backend: Separate GPU server or cloud service
```

### Mid-tier VPS (Lazy Loading)
```
CPU: 2-4 cores
RAM: 8GB+
Storage: 20GB
Cost: ~$20-40/month

Components:
- MCP server + Local TTS
- Nginx
- Model cache management
- Auto-unload after idle timeout
```

## Data Flow

### Proxy Mode Flow
```
1. LLM calls generate_speech(text, voice_id)
2. MCP server receives request
3. Load voice sample metadata from SQLite
4. Forward to remote backend via HTTP
5. Stream response back to LLM
6. Return audio bytes
```

**Latency**: Network overhead (~100-500ms) + Generation time

### Local Mode Flow
```
1. LLM calls generate_speech(text, voice_id)
2. MCP server receives request
3. Check if model loaded:
   - If not: Load model (5-10s first time)
   - If yes: Use cached model
4. Generate audio
5. Reset idle timeout
6. Return audio bytes
```

**Latency**: Model load (first time) or ~instant + Generation time

## Configuration File

```yaml
# mcp_config.yaml

mode: "proxy"  # proxy | lazy | always

# Proxy mode settings
proxy:
  backend_url: "https://gpu-server.example.com/api/tts"
  api_key: "${TTS_API_KEY}"
  timeout: 120
  max_retries: 3

# Local mode settings
local:
  device: "cpu"  # cpu | cuda | mps
  model_type: "english"  # english | multilingual
  lazy_unload_timeout: 600  # seconds (10 minutes)
  max_concurrent: 1

# Voice management
voices:
  storage: "local"  # local | s3
  local_path: "./voices"
  max_size_mb: 10

# Queue settings (optional)
queue:
  enabled: false
  backend: "memory"  # memory | redis
  redis_url: "redis://localhost:6379"
  max_queue_size: 10

# Server settings
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  log_level: "info"
```

## File Structure

```
chatterbox/
├── mcp_server/
│   ├── __init__.py
│   ├── server.py              # Main MCP server (lightweight)
│   ├── config.py              # Configuration loading
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py           # Abstract backend
│   │   ├── remote.py         # Proxy to remote service
│   │   └── local.py          # Local model (lazy/always)
│   ├── voice_manager.py      # Voice sample management
│   ├── queue.py              # Request queue
│   └── models.py             # Pydantic models
├── tts_backend/              # Separate process/server
│   ├── __init__.py
│   ├── api.py                # FastAPI server for inference
│   └── worker.py             # Background worker
├── docker/
│   ├── Dockerfile.mcp        # Lightweight MCP server
│   ├── Dockerfile.backend    # Heavy TTS backend
│   └── docker-compose.yml    # Full stack
└── deployment/
    ├── systemd/
    │   ├── chatterbox-mcp.service
    │   └── chatterbox-backend.service
    └── nginx/
        └── chatterbox.conf
```

## Memory Optimization Strategies

### 1. Model Quantization (Future)
- Use INT8 quantization for CPU inference
- Reduces model size from ~2GB to ~500MB
- Faster CPU inference

### 2. Streaming Responses
- Stream audio chunks back to client
- Don't hold full audio in memory
- Important for long-form content

### 3. Voice Sample Caching
- Cache prepared voice embeddings
- Reuse across requests (256-dim vector vs full audio)
- ~1KB per voice vs ~1MB audio file

### 4. Shared Model Loading
- Use separate FastAPI backend process
- MCP server(s) share one model instance
- Multiple MCP workers → 1 backend

## Cost Comparison

### Option A: Separate Servers
```
VPS (MCP):           $5/month   (512MB RAM)
GPU Server (Backend): $50/month  (RunPod, Vast.ai)
Total:               $55/month
```

### Option B: Single Larger VPS
```
VPS (8GB RAM):       $40/month  (DigitalOcean, Linode)
Total:               $40/month
(But slower CPU-only inference)
```

### Option C: Serverless
```
VPS (MCP):           $5/month
Modal/Replicate:     Pay per use (~$0.01/minute)
Total:               Variable
```

## Recommendation

**For most users**: Start with **Proxy Mode**
1. Deploy lightweight MCP server on cheap VPS ($5/mo)
2. Use existing Gradio endpoint or deploy backend separately
3. Easy to scale backend independently
4. Minimal VPS resource usage
5. Can migrate to local mode later if needed

**For development**: Use **Local Always-Loaded Mode**
- Fastest iteration
- No network dependency
- Good for testing

**For production with budget**: **Lazy Loading Mode**
- Balance between cost and performance
- Automatic resource management
- Good for moderate usage
