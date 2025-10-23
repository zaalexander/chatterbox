# Chatterbox TTS Repository Overview

## Project Summary
**Chatterbox** is an open-source, production-grade Text-To-Speech (TTS) system by Resemble AI that supports 23 languages. It's the first open-source TTS to offer emotion exaggeration control and robust multilingual zero-shot voice cloning. Licensed under MIT.

### Key Capabilities
- **Multilingual**: 23 languages (Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Turkish)
- **Zero-shot Voice Cloning**: Clone voices from reference audio without retraining
- **Emotion/Intensity Control**: Exaggeration parameter (0.25-2.0) for controlling speech expressiveness
- **Ultra-stable**: Alignment-informed inference
- **Watermarked**: Built-in Perth (Perceptual Threshold) watermarking for responsible AI
- **Fast**: Trained on 0.5M hours of cleaned data

---

## 1. Overall Architecture & Main Components

### System Architecture
The system uses a **two-stage pipeline**:

1. **Text-to-Token Generation (T3)**
   - Converts text to intermediate speech tokens
   - Uses Llama 3 (520M) as backbone
   - Condition-aware architecture with speaker embeddings and emotion control

2. **Token-to-Speech Generation (S3Gen)**
   - Converts speech tokens to mel-spectrograms
   - Applies flow matching/diffusion
   - Vocodes mel-spectrograms to waveform using HiFiGAN

### Main Components

```
src/chatterbox/
├── tts.py                    # English-only TTS class (ChatterboxTTS)
├── mtl_tts.py               # Multilingual TTS class (ChatterboxMultilingualTTS)
├── vc.py                    # Voice Conversion class (ChatterboxVC)
├── __init__.py              # Package exports
└── models/
    ├── t3/                  # Token-to-Token (Text→Speech Tokens)
    │   ├── t3.py           # Main T3 model
    │   ├── llama_configs.py # Llama 3 config (520M)
    │   ├── inference/       # Inference optimizations
    │   │   ├── t3_hf_backend.py
    │   │   └── alignment_stream_analyzer.py
    │   └── modules/
    │       ├── t3_config.py          # Configuration (English: 704 tokens, Multilingual: 2454 tokens)
    │       ├── cond_enc.py           # Conditioning encoder (speaker, emotion, etc.)
    │       ├── learned_pos_emb.py    # Position embeddings
    │       └── perceiver.py          # Perceiver resampler
    │
    ├── s3gen/               # Speech Synthesis Generator (Tokens→Speech)
    │   ├── s3gen.py        # Main S3Gen model
    │   ├── const.py        # Constants (S3GEN_SR = 24000)
    │   ├── configs.py      # CFM parameters
    │   ├── hifigan.py      # HiFiGAN vocoder
    │   ├── flow.py         # Flow matching architecture
    │   ├── flow_matching.py # Conditional Flow Matching
    │   ├── decoder.py      # Conditional decoder
    │   ├── f0_predictor.py # Fundamental frequency predictor
    │   ├── xvector.py      # Speaker embedding (CAMPPlus)
    │   ├── transformer/    # Conformer encoder
    │   │   ├── activation.py
    │   │   ├── attention.py
    │   │   ├── convolution.py
    │   │   ├── embedding.py
    │   │   ├── encoder_layer.py
    │   │   ├── positionwise_feed_forward.py
    │   │   ├── subsampling.py
    │   │   └── upsample_encoder.py
    │   ├── matcha/         # Matcha flow matching variant
    │   └── utils/
    │       └── mel.py      # Mel-spectrogram computation
    │
    ├── s3tokenizer/         # Speech Tokenization
    │   └── s3tokenizer.py  # S3TokenizerV2 wrapper (16kHz audio)
    │
    ├── tokenizers/          # Text Tokenization
    │   └── tokenizer.py    # English & Multilingual text tokenizers
    │
    ├── voice_encoder/       # Speaker Embedding Extraction
    │   ├── voice_encoder.py
    │   ├── config.py       # Voice encoder hyperparameters
    │   └── melspec.py      # Mel-spectrogram computation
    │
    └── utils.py            # Utility classes
```

---

## 2. Core TTS Functionality Files

### Main TTS Classes

#### **ChatterboxTTS (English-only)**
- **File**: `/home/user/chatterbox/src/chatterbox/tts.py` (271 lines)
- **Key Methods**:
  - `from_pretrained(device)`: Downloads and loads pre-trained English model
  - `from_local(ckpt_dir, device)`: Load from local checkpoint
  - `prepare_conditionals(wav_fpath, exaggeration)`: Prepare speaker embedding & conditioning
  - `generate(text, audio_prompt_path, exaggeration, cfg_weight, temperature, ...)`: Synthesize speech
- **Models Loaded**:
  - `VoiceEncoder (ve)`: Speaker embedding extraction
  - `T3`: Text-to-token generation
  - `S3Gen`: Token-to-speech generation
  - `EnTokenizer`: English text tokenization
- **Special Features**:
  - Watermarking via `perth.PerthImplicitWatermarker()`
  - Conditional training with speaker embeddings and emotion control
  - Sampling rate: 24000 Hz (S3GEN_SR)

#### **ChatterboxMultilingualTTS**
- **File**: `/home/user/chatterbox/src/chatterbox/mtl_tts.py` (301 lines)
- **Supports**: 23 languages (defined in `SUPPORTED_LANGUAGES` dict)
- **Key Differences**:
  - Uses `T3Config.multilingual()` with 2454 text tokens (vs 704 for English)
  - Uses `MTLTokenizer` for multilingual text tokenization
  - Downloads from HuggingFace with `snapshot_download()`
  - Language-specific text normalization (e.g., Japanese kanji→hiragana, Russian stress markers)
- **Same Generation Pipeline**: prepare_conditionals → generate with language_id parameter

#### **ChatterboxVC (Voice Conversion)**
- **File**: `/home/user/chatterbox/src/chatterbox/vc.py` (104 lines)
- **Purpose**: Convert voice from one speaker to another
- **Key Methods**:
  - `set_target_voice(wav_fpath)`: Set target voice characteristics
  - `generate(audio, target_voice_path)`: Convert audio to target voice
- **Models Used**: Only `S3Gen` (no T3 needed)

### Text Tokenization

#### **English Tokenizer (EnTokenizer)**
- **File**: `/home/user/chatterbox/src/chatterbox/models/tokenizers/tokenizer.py`
- **Uses**: HuggingFace `tokenizers.Tokenizer`
- **Vocab**: 704 tokens
- **Special Tokens**: `[START]`, `[STOP]`, `[UNK]`, `[SPACE]`, `[PAD]`, `[SEP]`, `[CLS]`, `[MASK]`
- **Processing**: Text → replace spaces with `[SPACE]` → encode

#### **Multilingual Tokenizer (MTLTokenizer)**
- Supports grapheme-based tokenization for all 23 languages
- Language-specific processing:
  - **Japanese**: Kanji→hiragana conversion via `pykakasi`
  - **Russian**: Stress marker addition via `russian-text-stresser`
  - **Chinese**: Pinyin conversion via `spacy-pkuseg`

### Speech Tokenization

#### **S3Tokenizer (S3TokenizerV2)**
- **File**: `/home/user/chatterbox/src/chatterbox/models/s3tokenizer/s3tokenizer.py`
- **Purpose**: Convert audio waveforms to discrete speech tokens
- **Config**:
  - Input SR: 16,000 Hz
  - Token rate: 25 tokens/second
  - Token hop: 640 samples (160ms @ 16kHz = 0.16s)
  - Vocab size: 6,561 tokens (valid), 6,562 is stop token
- **Methods**:
  - `forward(wavs)`: Convert waveforms to token sequences
  - `pad()`: Pad audio to align with token boundaries

### Voice Encoder (Speaker Embedding)

#### **VoiceEncoder**
- **File**: `/home/user/chatterbox/src/chatterbox/models/voice_encoder/voice_encoder.py`
- **Purpose**: Extract fixed-size speaker embeddings from reference audio
- **Config**:
  - Sample rate: 16,000 Hz
  - Mel spectrogram: 40 bands, 400-point STFT, 160-point hop
  - Speaker embedding size: 256-dim
  - Uses LSTM backbone
- **Key Method**: `embeds_from_wavs(wavs, sample_rate)` → (batch, 256)

---

## 3. Dependencies & Models Used

### Core Dependencies (from pyproject.toml)

```
numpy>=1.24.0,<1.26.0
librosa==0.11.0              # Audio processing
s3tokenizer                  # Speech tokenization
torch==2.6.0
torchaudio==2.6.0
transformers==4.46.3         # Llama 3 models
diffusers==0.29.0            # Diffusion models
resemble-perth==1.0.1        # Watermarking
conformer==0.3.2             # Encoder architecture
safetensors==0.5.3           # Model serialization
spacy-pkuseg                 # Chinese text processing
pykakasi==2.3.0              # Japanese kanji conversion
gradio==5.44.1               # Web UI
russian-text-stresser        # Russian stress marking (from git)
```

### Pre-trained Models

#### **Model Checkpoints** (downloaded from HuggingFace)
- **Repository**: `ResembleAI/chatterbox`
- **English Model Files**:
  - `ve.safetensors`: Voice encoder weights
  - `t3_cfg.safetensors`: T3 model (394 lines)
  - `s3gen.safetensors`: S3Gen model (298 lines)
  - `tokenizer.json`: English tokenizer vocab
  - `conds.pt`: Built-in default voice conditioning

- **Multilingual Model Files**:
  - `ve.pt`: Voice encoder (PyTorch format)
  - `t3_mtl23ls_v2.safetensors`: Multilingual T3 (23 languages)
  - `s3gen.pt`: S3Gen model
  - `grapheme_mtl_merged_expanded_v1.json`: Multilingual tokenizer
  - `Cangjie5_TC.json`: Traditional Chinese conversion
  - `conds.pt`: Default voice conditioning

#### **Model Architecture Details**

**T3 (Text-to-Token) Model**:
- Backbone: Llama 3 (520M parameters)
- Hidden size: 1,024
- Attention heads: 16
- Layers: 30
- Max position embeddings: 131,072
- Text token vocab size: 704 (English) / 2,454 (Multilingual)
- Speech token vocab size: 8,194
- Output: 1,000 speech tokens maximum per generation

**S3Gen (Token-to-Speech) Model**:
- Input: Speech tokens (6,561 vocab)
- Output: Waveform (24 kHz)
- Components:
  - **Encoder**: UpsampleConformerEncoder (6 blocks, 512 output size)
  - **Flow Matching**: CausalConditionalCFM with cosine scheduler
  - **Decoder**: ConditionalDecoder (80 mel bands, 4 main + 12 mid blocks)
  - **Vocoder**: HiFiGAN variant (HiFTGenerator)
- Speaker embedding: 80-dim from CAMPPlus (X-Vector variant)

**Voice Encoder**:
- Backbone: LSTM-based speaker verification model
- Output: 256-dim speaker embeddings
- Based on: Real-Time-Voice-Cloning project

### Acknowledgements/References
- **CosyVoice**: Flow matching and vocoder architecture
- **Real-Time-Voice-Cloning**: Voice encoder architecture
- **HiFT-GAN**: Vocoder improvements
- **Llama 3**: Foundation model backbone
- **S3Tokenizer**: Speech tokenization

---

## 4. Existing API & Server Implementations

### Gradio Web Applications

#### **1. English TTS App** (`gradio_tts_app.py`)
- **Purpose**: English-only TTS with voice cloning
- **Interface**:
  - Text input (max 300 chars)
  - Reference audio upload/microphone/URL
  - Exaggeration slider (0.25-2.0)
  - CFG/Pace slider (0.0-1.0)
  - Temperature, min_p, top_p, repetition_penalty (advanced options)
  - Random seed control
- **Features**:
  - Model loads once per session
  - Queue support (max 50, concurrency 1)
  - Seed control for reproducibility

#### **2. Multilingual TTS App** (`multilingual_app.py`)
- **Purpose**: Multilingual TTS (23 languages) with reference audio
- **Interface**:
  - Language selector (dropdown with 23 options)
  - Text input
  - Language-specific default audio prompts & examples
  - Reference audio (optional, language-matched)
  - Exaggeration, temperature, CFG weight controls
  - Random seed
- **Features**:
  - Dynamic language change updates defaults
  - Language-specific tips & note about accent matching
  - Uses `mcp_server=True` (MCP protocol support)
  - Supports both upload and microphone input

#### **3. Voice Conversion App** (`gradio_vc_app.py`)
- **Purpose**: Convert voice from one speaker to another
- **Interface**:
  - Input audio (upload/microphone)
  - Target voice audio (optional, defaults to built-in)
  - Output audio playback
- **Lightweight**: Only requires S3Gen model

### Command-Line Examples

#### **`example_tts.py`**: English and Multilingual TTS
```python
# English generation
model = ChatterboxTTS.from_pretrained(device="cuda")
wav = model.generate(text)
ta.save("output.wav", wav, model.sr)

# Multilingual generation
multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device)
wav = multilingual_model.generate(text, language_id="fr")
ta.save("output.wav", wav, model.sr)

# With voice cloning
wav = model.generate(text, audio_prompt_path="reference.wav")
```

#### **`example_vc.py`**: Voice Conversion
```python
model = ChatterboxVC.from_pretrained(device="cuda")
wav = model.generate(audio="input.wav", target_voice_path="target.wav")
ta.save("output.wav", wav, model.sr)
```

#### **`example_for_mac.py`**: macOS Optimized
- Patches `torch.load()` to default to correct device (MPS/CPU)
- Sets `exaggeration=2.0` for more dramatic speech
- Higher `cfg_weight=0.5` for stability

### No REST API/FastAPI Currently
- **Note**: The codebase has no built-in FastAPI/REST server
- Gradio is the primary web interface
- Could be extended with FastAPI wrapper for production deployment

---

## 5. Configuration Files & Setup Requirements

### Dependencies Configuration

#### **pyproject.toml**
- Project: `chatterbox-tts` v0.1.4
- Python: >=3.10
- Package source: `src/` directory layout
- Pinned versions for reproducibility
- Install: `pip install -e .` (editable mode)

### Model Configuration Files

#### **T3Config** (`models/t3/modules/t3_config.py`)
```python
# English-only variant
text_tokens_dict_size = 704
text_tokens_dict_size_multilingual = 2454
max_text_tokens = 2048
speech_tokens_dict_size = 8194
max_speech_tokens = 4096
llama_config_name = "Llama_520M"
input_pos_emb = "learned"
speech_cond_prompt_len = 150
emotion_adv = True  # Enable emotion/exaggeration control
use_perceiver_resampler = True
```

#### **S3Gen Config** (`models/s3gen/configs.py`)
```python
CFM_PARAMS = {
    "sigma_min": 1e-06,
    "solver": "euler",
    "t_scheduler": "cosine",
    "training_cfg_rate": 0.2,
    "inference_cfg_rate": 0.7,
    "reg_loss_type": "l1"
}
```

#### **VoiceEncoder Config** (`models/voice_encoder/config.py`)
```python
num_mels = 40
sample_rate = 16000
speaker_embed_size = 256
ve_hidden_size = 256
n_fft = 400
hop_size = 160
fmax = 8000
```

#### **Llama Config** (`models/t3/llama_configs.py`)
```python
# Llama 3 520M configuration
vocab_size = 8  # Custom embeddings
hidden_size = 1024
num_hidden_layers = 30
num_attention_heads = 16
head_dim = 64
max_position_embeddings = 131072
rope_scaling = {"factor": 8.0, ...}
```

### Audio Processing Constants

```python
# S3Tokenizer (Audio → Tokens)
S3_SR = 16_000           # Input sample rate
S3_HOP = 160             # Hop size (160 samples)
S3_TOKEN_HOP = 640       # Token hop (25 tokens/sec)
S3_TOKEN_RATE = 25       # Token frequency
SPEECH_VOCAB_SIZE = 6561 # Valid tokens

# S3Gen (Tokens → Waveform)
S3GEN_SR = 24_000        # Output sample rate
```

### Watermarking Configuration

```python
# Perth watermarker (embedded in all outputs)
watermarker = perth.PerthImplicitWatermarker()
# Survives: MP3 compression, audio editing, common manipulations
# Detection accuracy: ~100%
```

---

## 6. Documentation & Usage Requirements

### Installation Methods

#### **From PyPI (Recommended)**
```bash
pip install chatterbox-tts
```

#### **From Source (Development)**
```bash
git clone https://github.com/resemble-ai/chatterbox.git
cd chatterbox
pip install -e .
```

#### **Requirements**
- Python 3.10+ (developed/tested on 3.11)
- CUDA 11.x+ (optional, for GPU acceleration)
- macOS 12.3+ with M-series chip (for MPS support)
- ~8GB RAM minimum for inference
- ~2GB for model downloads

### Key Usage Parameters

#### **Text Normalization**
- Auto-capitalizes first letter
- Removes extra spaces
- Replaces uncommon punctuation (…→, ; →,, etc.)
- Adds period if no ending punctuation

#### **Generation Parameters**

| Parameter | Range | Default | Purpose |
|-----------|-------|---------|---------|
| `exaggeration` | 0.25-2.0 | 0.5 | Emotional intensity (0.5=neutral) |
| `cfg_weight` | 0.0-1.0 | 0.5 | Classifier-free guidance weight |
| `temperature` | 0.05-5.0 | 0.8 | Sampling temperature (randomness) |
| `min_p` | 0.0-1.0 | 0.05 | Minimum probability threshold |
| `top_p` | 0.0-1.0 | 1.0 | Nucleus sampling (1.0 = disabled) |
| `repetition_penalty` | 1.0-2.0 | 1.2 | Reduces token repetition |

#### **Language-Specific Tips**
- **General Use**: exaggeration=0.5, cfg_weight=0.5
- **Expressive Speech**: exaggeration=0.7+, cfg_weight=0.3 (slower pacing)
- **Language Transfer**: Set cfg_weight=0 to prevent accent transfer
- **Fast Speech**: Lower cfg_weight to ~0.3 if reference is fast

### Model Download Details

#### **First Run Behavior**
```python
model = ChatterboxTTS.from_pretrained(device="cuda")
# Downloads ~2GB files from HuggingFace automatically
# Cached in: ~/.cache/huggingface/hub/
# Files: ve.safetensors, t3_cfg.safetensors, s3gen.safetensors, tokenizer.json, conds.pt
```

#### **Supported Languages (Multilingual Model)**
23 languages: Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Turkish

### Watermark Detection

```python
import perth
import librosa

AUDIO_PATH = "generated.wav"
watermarked_audio, sr = librosa.load(AUDIO_PATH, sr=None)
watermarker = perth.PerthImplicitWatermarker()
watermark = watermarker.get_watermark(watermarked_audio, sample_rate=sr)
# Output: 0.0 (no watermark) or 1.0 (watermarked)
```

### Platform-Specific Notes

#### **Linux/Windows**
- Uses CUDA if available
- Falls back to CPU otherwise
- Best performance with GPU acceleration

#### **macOS**
- M1/M2/M3/M4 chip: Use `device="mps"` for Metal acceleration
- Intel Mac: Use `device="cpu"`
- `example_for_mac.py` patches torch.load() for device handling

#### **Docker**
- No official Dockerfile (could be added)
- `.gitignore` ignores checkpoints directory
- Generated `.wav` files are ignored (output only)

---

## 7. System Information

### Repository Structure
```
chatterbox/
├── src/chatterbox/           # Main package
├── pyproject.toml            # Project config
├── README.md                 # Documentation
├── LICENSE                   # MIT License
├── example_tts.py           # Basic TTS example
├── example_vc.py            # Voice conversion example
├── example_for_mac.py       # macOS-optimized example
├── gradio_tts_app.py        # English TTS web app
├── gradio_vc_app.py         # Voice conversion web app
├── multilingual_app.py      # Multilingual TTS web app
└── .gitignore               # Standard Python ignores
```

### Version Info
- **Version**: 0.1.4
- **Python**: 3.10+
- **License**: MIT
- **Repository**: https://github.com/resemble-ai/chatterbox

### File Statistics
- **T3 Model**: 394 lines
- **S3Gen Model**: 298 lines
- **Main TTS Class**: 271 lines
- **Multilingual TTS Class**: 301 lines
- **Total Python**: 50+ files across models

