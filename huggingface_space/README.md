---
title: Chatterbox TTS
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
python_version: 3.11
---

# Chatterbox TTS 🎙️

Production-grade text-to-speech with zero-shot voice cloning and emotion control.

## Features

- **🗣️ Zero-shot Voice Cloning**: Clone any voice from a short sample (5-10 seconds)
- **🎭 Emotion Control**: Adjust speech expressiveness from neutral (0.5) to highly dramatic (2.0)
- **🎵 High-quality Audio**: 24kHz output with natural prosody
- **🔒 Watermarked**: All outputs include Perth imperceptible watermarks for responsible AI
- **⚡ Fast**: GPU-accelerated inference (~3-5s per generation)

## Usage

1. **Enter text** in the text box
2. **Optional**: Upload a voice sample for voice cloning
3. **Adjust settings** (exaggeration, pacing, temperature)
4. **Click Generate** and wait for your audio!

## Voice Cloning Tips

For best voice cloning results:
- Use 5-10 second clear audio samples
- Minimal background noise
- Clear, natural speech
- Match language of sample to target language

## API Access

This Space supports programmatic access via the Gradio API:

```python
from gradio_client import Client

client = Client("YOUR_USERNAME/chatterbox-tts")

result = client.predict(
    text="Hello, this is a test!",
    audio_prompt=None,  # Or path to voice sample
    language="English",
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8,
    seed=0,
    api_name="/generate"
)

# result is (sample_rate, audio_array)
```

## Use as MCP Server Backend

This Space can be used as a free backend for the Chatterbox MCP server:

```yaml
# mcp_config.yaml
mode: "proxy"
proxy:
  backend_url: "https://YOUR_USERNAME-chatterbox-tts.hf.space"
```

See the [MCP Server documentation](https://github.com/resemble-ai/chatterbox/blob/main/mcp_server/README.md) for details.

## Links

- **GitHub**: https://github.com/resemble-ai/chatterbox
- **Resemble AI**: https://resemble.ai
- **Documentation**: https://github.com/resemble-ai/chatterbox#readme

## Disclaimer

This is a powerful voice cloning tool. Please use responsibly:
- ✅ Create content you have permission for
- ✅ Respect voice ownership and consent
- ✅ Follow local laws and regulations
- ❌ Don't create misleading or harmful content
- ❌ Don't impersonate others without permission

All outputs are watermarked with Perth for authenticity verification.
