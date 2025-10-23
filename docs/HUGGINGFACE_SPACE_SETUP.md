# HuggingFace Space Backend Setup Guide

Complete guide to deploying Chatterbox TTS as a free HuggingFace Space and using it as a backend for your lightweight MCP server.

## Overview

This setup gives you:
- **Free TTS backend** (HuggingFace Space with GPU)
- **$5/month VPS** (running lightweight MCP server)
- **Total cost: $5/month** (vs $50+ for GPU server)

```
┌─────────────┐      ┌──────────────────┐      ┌───────────────────────┐
│ LLM Agent   │─────▶│ MCP Server       │─────▶│ HuggingFace Space     │
│ (Claude)    │      │ (VPS: $5/mo)     │      │ (FREE GPU!)           │
│             │      │ ~100MB RAM       │      │ Gradio App            │
└─────────────┘      └──────────────────┘      └───────────────────────┘
```

## Part 1: Deploy Chatterbox to HuggingFace Space

### Step 1: Create HuggingFace Account

1. Go to https://huggingface.co/join
2. Sign up for free account
3. Verify your email

### Step 2: Create New Space

1. Go to https://huggingface.co/new-space
2. Fill in details:
   - **Space name:** `chatterbox-tts` (or your choice)
   - **License:** MIT
   - **Select SDK:** Gradio
   - **Hardware:** CPU basic (free) or upgrade to GPU (see pricing below)
   - **Visibility:** Public (free) or Private (requires subscription)

3. Click **Create Space**

### Step 3: Prepare Files for Space

Create these files in your HuggingFace Space repository:

#### `app.py` (Main Gradio app)

```python
import gradio as gr
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Detect device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load model (choose one)
# Option 1: English only
model = ChatterboxTTS.from_pretrained(device=DEVICE)

# Option 2: Multilingual (23 languages)
# model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)

def generate_tts(text, audio_prompt, exaggeration, cfg_weight, temperature):
    """Generate TTS audio."""
    if not text:
        return None

    try:
        wav = model.generate(
            text,
            audio_prompt_path=audio_prompt if audio_prompt else None,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )

        # Return as (sample_rate, audio_array) for Gradio
        return (model.sr, wav.squeeze(0).numpy())

    except Exception as e:
        print(f"Error: {e}")
        return None


# Create Gradio interface
with gr.Blocks(title="Chatterbox TTS") as demo:
    gr.Markdown("# Chatterbox TTS - Production-grade Voice Cloning")
    gr.Markdown(
        "Generate natural speech with emotion control and voice cloning. "
        "[GitHub](https://github.com/resemble-ai/chatterbox)"
    )

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Text to synthesize",
                placeholder="Enter text here...",
                lines=5,
                max_lines=10,
            )

            audio_prompt = gr.Audio(
                label="Voice Sample (optional - for voice cloning)",
                sources=["upload", "microphone"],
                type="filepath"
            )

            with gr.Accordion("Advanced Settings", open=False):
                exaggeration = gr.Slider(
                    minimum=0.25,
                    maximum=2.0,
                    value=0.5,
                    step=0.05,
                    label="Exaggeration (emotion intensity)"
                )
                cfg_weight = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="CFG Weight (pacing control)"
                )
                temperature = gr.Slider(
                    minimum=0.05,
                    maximum=2.0,
                    value=0.8,
                    step=0.05,
                    label="Temperature"
                )

            generate_btn = gr.Button("Generate Speech", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Generated Speech")

            gr.Examples(
                examples=[
                    ["Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."],
                    ["The quick brown fox jumps over the lazy dog, demonstrating the full range of phonetic sounds."],
                    ["Welcome to the future of text-to-speech technology. With emotion control and voice cloning, the possibilities are endless."],
                ],
                inputs=[text_input],
            )

    generate_btn.click(
        fn=generate_tts,
        inputs=[text_input, audio_prompt, exaggeration, cfg_weight, temperature],
        outputs=audio_output,
    )

# Launch app
if __name__ == "__main__":
    demo.queue(max_size=10).launch()
```

#### `requirements.txt`

```txt
chatterbox-tts
gradio>=4.0.0
torch>=2.0.0
torchaudio>=2.0.0
```

#### `README.md`

```markdown
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
---

# Chatterbox TTS

Production-grade text-to-speech with voice cloning and emotion control.

- **Zero-shot voice cloning** from short samples
- **Emotion/exaggeration control** (0.25-2.0)
- **High-quality 24kHz audio**
- **Watermarked outputs** for responsible AI

[GitHub Repository](https://github.com/resemble-ai/chatterbox)
```

### Step 4: Upload to HuggingFace Space

**Option A: Using Git (Recommended)**

```bash
# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/chatterbox-tts
cd chatterbox-tts

# Add files
cp /path/to/app.py .
cp /path/to/requirements.txt .
cp /path/to/README.md .

# Commit and push
git add .
git commit -m "Initial Chatterbox TTS deployment"
git push
```

**Option B: Using Web Interface**

1. Go to your Space page
2. Click **Files** tab
3. Click **Add file** → **Create a new file**
4. Copy/paste content for each file above
5. Commit changes

### Step 5: Wait for Build

1. HuggingFace will automatically build your Space
2. Watch the **Logs** tab for progress
3. First build takes ~5-10 minutes (downloading models)
4. Status will change to **Running** when ready

### Step 6: Test Your Space

1. Open your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/chatterbox-tts`
2. Enter test text: "Hello, this is a test"
3. Click **Generate Speech**
4. Verify audio plays correctly

**Your free TTS backend is now live!** 🎉

---

## Part 2: Configure MCP Server to Use HuggingFace Space

### Step 1: Get Your Space URL

Your Space URL format:
```
https://YOUR_USERNAME-chatterbox-tts.hf.space
```

Example:
```
https://john-chatterbox-tts.hf.space
```

### Step 2: Configure MCP Server

Edit `mcp_config.yaml`:

```yaml
mode: "proxy"

proxy:
  # Use your HuggingFace Space URL
  backend_url: "https://YOUR_USERNAME-chatterbox-tts.hf.space"

  # No API key needed for public Spaces
  api_key: null

  # Increase timeout (HF Spaces can be slow on CPU)
  timeout: 180

  max_retries: 3

voices:
  storage: "local"
  local_path: "./voices"
  max_size_mb: 10

server:
  host: "0.0.0.0"
  port: 8000
  log_level: "info"
```

### Step 3: Deploy MCP Server on VPS

**Using Docker:**

```bash
# On your VPS
git clone <your-repo>
cd chatterbox

# Build lightweight MCP server image
docker build -f Dockerfile.mcp -t chatterbox-mcp .

# Run with environment variable
docker run -d \
  --name chatterbox-mcp \
  -p 8000:8000 \
  -v $(pwd)/voices:/app/voices \
  -v $(pwd)/mcp_config.yaml:/app/mcp_config.yaml \
  -e TTS_BACKEND_URL=https://YOUR_USERNAME-chatterbox-tts.hf.space \
  --restart unless-stopped \
  chatterbox-mcp

# Check logs
docker logs -f chatterbox-mcp
```

**Using Python directly:**

```bash
# Install minimal dependencies
pip install mcp httpx pydantic pyyaml

# Run server
python -m mcp_server.server
```

### Step 4: Test End-to-End

```bash
# From another terminal, test the MCP server
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello from MCP server via HuggingFace Space!",
    "exaggeration": 0.6
  }'
```

---

## Part 3: Optimization Tips

### GPU Acceleration (Optional)

**Free Tier:**
- HuggingFace offers **free CPU Spaces** (slow but free)
- Generation time: ~30-60s for medium text

**Upgrade to GPU:**
- **Pricing:** ~$0.60/hour (billed per second)
- **GPU options:** T4, A10G, A100
- **Speed:** 10-20x faster than CPU
- **How to upgrade:**
  1. Go to Space Settings
  2. Change Hardware → Select GPU
  3. Confirm billing

**Cost estimate with GPU:**
```
Average usage: 10 hours/month
Cost: 10 hours × $0.60 = $6/month

Total: $5 VPS + $6 GPU = $11/month
(Still much cheaper than dedicated GPU server!)
```

### Reducing Cold Start Time

HuggingFace Spaces go to sleep after inactivity. To keep warm:

**Option 1: Persistent mode (Paid)**
- Settings → Sleep time → Never sleep
- Requires PRO subscription ($9/month)

**Option 2: Keep-alive ping**

Create a simple cron job on your VPS:

```bash
# Add to crontab: crontab -e
*/10 * * * * curl -s https://YOUR_USERNAME-chatterbox-tts.hf.space > /dev/null
```

This pings your Space every 10 minutes to prevent sleep.

**Option 3: Smart retry in MCP server**

The MCP server already handles retries. First request after sleep takes ~30s (model loading), subsequent requests are fast.

### Handling Rate Limits

Public Spaces have rate limits:
- ~100 requests/hour for free tier
- Upgrade to PRO for higher limits

If you hit limits:
1. Deploy your own GPU server
2. Use paid HF Space tier
3. Implement request queueing in MCP server

---

## Part 4: Advanced Configurations

### Private Space (for sensitive data)

```yaml
# In Space README.md front matter:
---
private: true
---
```

Then add authentication to MCP config:

```yaml
proxy:
  backend_url: "https://YOUR_USERNAME-chatterbox-tts.hf.space"
  api_key: "${HF_TOKEN}"  # Your HuggingFace token
```

Get token from: https://huggingface.co/settings/tokens

### Multilingual Space

Update `app.py` to use multilingual model:

```python
# Load multilingual model
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)

# Add language selector to interface
language = gr.Dropdown(
    choices=["en", "fr", "es", "de", "zh", "ja", "ko", ...],
    value="en",
    label="Language"
)

# Update generate function
def generate_tts(text, language, audio_prompt, exaggeration, cfg_weight, temperature):
    wav = model.generate(
        text,
        language_id=language,
        audio_prompt_path=audio_prompt,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
    )
    return (model.sr, wav.squeeze(0).numpy())
```

### Custom Domain

HuggingFace PRO subscribers can use custom domains:

1. Settings → Custom Domain
2. Add CNAME record: `tts.yourdomain.com → YOUR_USERNAME-chatterbox-tts.hf.space`

Then update MCP config:
```yaml
proxy:
  backend_url: "https://tts.yourdomain.com"
```

---

## Part 5: Troubleshooting

### Space won't start

**Check logs:**
1. Go to Space page
2. Click **Logs** tab
3. Look for errors

**Common issues:**

**1. Out of memory**
```
RuntimeError: CUDA out of memory
```
**Solution:** Upgrade to larger GPU or use CPU

**2. Dependencies failed**
```
ERROR: Could not find a version that satisfies...
```
**Solution:** Pin versions in requirements.txt:
```txt
chatterbox-tts==0.1.4
torch==2.6.0
```

**3. Model download timeout**
```
TimeoutError: Model download timed out
```
**Solution:** HF will retry automatically, wait 5-10 minutes

### MCP server can't reach Space

**Test connection:**
```bash
curl https://YOUR_USERNAME-chatterbox-tts.hf.space
```

**If fails:**
1. Check Space status (should be "Running")
2. Verify URL is correct
3. Check firewall on VPS

**If succeeds but MCP fails:**
1. Check MCP server logs: `docker logs chatterbox-mcp`
2. Verify `mcp_config.yaml` backend_url
3. Increase timeout to 300s

### Slow generation times

**CPU Space (free tier):**
- Expected: 30-60s for medium text
- Solution: Upgrade to GPU ($0.60/hr)

**GPU Space but still slow:**
1. Check Space isn't sleeping (first request is slower)
2. Verify GPU is actually allocated (check logs)
3. Reduce text length or use chunking

### Space keeps sleeping

**Symptoms:**
- First request very slow (~30s)
- Subsequent requests fast

**Solutions:**
1. Upgrade to PRO + persistent mode ($9/mo)
2. Use keep-alive ping (cron job)
3. Accept the tradeoff (free but occasional delays)

---

## Part 6: Cost Comparison

### Setup A: Free (but slow)
```
VPS (MCP):        $5/month
HF Space (CPU):   FREE
─────────────────────────
Total:            $5/month
Speed:            30-60s per generation
```

### Setup B: Fast & affordable
```
VPS (MCP):        $5/month
HF Space (GPU):   $6/month (10 hours usage)
─────────────────────────
Total:            $11/month
Speed:            3-5s per generation
```

### Setup C: Pro (always-on)
```
VPS (MCP):        $5/month
HF PRO:           $9/month (includes persistent Space)
HF Space (GPU):   $6/month (10 hours usage)
─────────────────────────
Total:            $20/month
Speed:            3-5s, no cold starts
```

### vs. Self-hosted GPU
```
GPU Server:       $50-100/month (RunPod, Vast.ai)
Speed:            2-3s per generation
```

**Recommendation:** Start with Setup A (free CPU), upgrade to Setup B if you need speed.

---

## Summary

✅ **HuggingFace Space as free TTS backend**
✅ **Lightweight MCP server on $5 VPS**
✅ **Total cost: $5-20/month** (vs $50+ self-hosted)
✅ **Easy deployment with Gradio**
✅ **Optional GPU upgrade for speed**
✅ **No infrastructure management**

**Next steps:**
1. Create HuggingFace account
2. Deploy Chatterbox Space (5-10 minutes)
3. Configure MCP server
4. Test with LLM agent
5. Optionally upgrade to GPU for speed

🎉 You now have a production-ready TTS system for $5/month!
