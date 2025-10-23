"""
Chatterbox TTS - HuggingFace Space Application

Deploy this to HuggingFace Spaces for a free TTS backend.
Works with the lightweight MCP server in proxy mode.

Instructions:
1. Create new Space on HuggingFace
2. Upload this file as app.py
3. Upload requirements.txt
4. Upload README.md with Space metadata
5. Wait for build (~5-10 minutes)
6. Use Space URL in MCP server config
"""

import gradio as gr
import torch
import numpy as np
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Configuration
USE_MULTILINGUAL = False  # Set to True for 23-language support
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Loading Chatterbox TTS...")
print(f"   Device: {DEVICE}")
print(f"   Multilingual: {USE_MULTILINGUAL}")

# Load model
if USE_MULTILINGUAL:
    model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
    SUPPORTED_LANGUAGES = {
        "Arabic": "ar",
        "Chinese": "zh",
        "Danish": "da",
        "Dutch": "nl",
        "English": "en",
        "Finnish": "fi",
        "French": "fr",
        "German": "de",
        "Greek": "el",
        "Hebrew": "he",
        "Hindi": "hi",
        "Italian": "it",
        "Japanese": "ja",
        "Korean": "ko",
        "Malay": "ms",
        "Norwegian": "no",
        "Polish": "pl",
        "Portuguese": "pt",
        "Russian": "ru",
        "Spanish": "es",
        "Swahili": "sw",
        "Swedish": "sv",
        "Turkish": "tr",
    }
else:
    model = ChatterboxTTS.from_pretrained(device=DEVICE)
    SUPPORTED_LANGUAGES = None

print(f"✅ Model loaded successfully!")


def generate_tts(
    text,
    audio_prompt,
    language,
    exaggeration,
    cfg_weight,
    temperature,
    seed
):
    """Generate TTS audio."""
    if not text or len(text.strip()) == 0:
        gr.Warning("Please enter some text to synthesize")
        return None

    if len(text) > 500:
        gr.Warning("Text is long, this may take a while...")

    try:
        # Set seed for reproducibility
        if seed > 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Prepare generation kwargs
        gen_kwargs = {
            "text": text,
            "audio_prompt_path": audio_prompt if audio_prompt else None,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
        }

        # Add language for multilingual model
        if USE_MULTILINGUAL and language:
            gen_kwargs["language_id"] = SUPPORTED_LANGUAGES.get(language, "en")

        # Generate audio
        wav = model.generate(**gen_kwargs)

        # Convert to numpy for Gradio
        audio_np = wav.squeeze(0).numpy()

        return (model.sr, audio_np)

    except Exception as e:
        gr.Error(f"Generation failed: {str(e)}")
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# Create Gradio interface
with gr.Blocks(
    title="Chatterbox TTS",
    theme=gr.themes.Soft(),
    css="""
    .header {text-align: center; margin-bottom: 1em;}
    .footer {text-align: center; margin-top: 2em; font-size: 0.9em; color: #666;}
    """
) as demo:
    # Header
    gr.HTML("""
        <div class="header">
            <h1>🎙️ Chatterbox TTS</h1>
            <p>Production-grade Text-to-Speech with Voice Cloning & Emotion Control</p>
            <p>
                <a href="https://github.com/resemble-ai/chatterbox" target="_blank">GitHub</a> •
                <a href="https://resemble.ai" target="_blank">Resemble AI</a>
            </p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            # Text input
            text_input = gr.Textbox(
                label="📝 Text to Synthesize",
                placeholder="Enter your text here...\n\nTip: For best results, use clear punctuation and keep sentences concise.",
                lines=6,
                max_lines=15,
            )

            # Voice cloning
            audio_prompt = gr.Audio(
                label="🎤 Voice Sample (Optional)",
                sources=["upload", "microphone"],
                type="filepath",
                info="Upload a voice sample (5-10 seconds recommended) for voice cloning. Leave empty for default voice."
            )

            # Language selector (only for multilingual)
            if USE_MULTILINGUAL:
                language = gr.Dropdown(
                    choices=list(SUPPORTED_LANGUAGES.keys()),
                    value="English",
                    label="🌍 Language",
                    info="Select the language of your text"
                )
            else:
                language = gr.State(None)

            # Advanced settings
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                exaggeration = gr.Slider(
                    minimum=0.25,
                    maximum=2.0,
                    value=0.5,
                    step=0.05,
                    label="Exaggeration (Emotion Intensity)",
                    info="0.5 = neutral, higher = more expressive"
                )

                cfg_weight = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="CFG Weight (Pacing Control)",
                    info="Lower = faster speech, higher = slower pacing"
                )

                temperature = gr.Slider(
                    minimum=0.05,
                    maximum=2.0,
                    value=0.8,
                    step=0.05,
                    label="Temperature",
                    info="Controls randomness in generation"
                )

                seed = gr.Number(
                    value=0,
                    label="Random Seed",
                    info="0 = random, >0 = reproducible generation",
                    precision=0
                )

            # Generate button
            generate_btn = gr.Button(
                "🎵 Generate Speech",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=1):
            # Output
            audio_output = gr.Audio(
                label="🔊 Generated Speech",
                type="numpy",
                show_download_button=True
            )

            # Examples
            gr.Examples(
                examples=[
                    [
                        "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill.",
                        None,
                        "English" if USE_MULTILINGUAL else None,
                        0.5,
                        0.5,
                        0.8,
                        0
                    ],
                    [
                        "The quick brown fox jumps over the lazy dog, demonstrating the full range of phonetic sounds in English speech.",
                        None,
                        "English" if USE_MULTILINGUAL else None,
                        0.4,
                        0.5,
                        0.8,
                        0
                    ],
                    [
                        "Welcome to the future of text-to-speech technology! With emotion control and voice cloning, the possibilities are truly endless.",
                        None,
                        "English" if USE_MULTILINGUAL else None,
                        0.7,
                        0.3,
                        0.8,
                        0
                    ],
                ] + ([
                    [
                        "Bonjour! Comment allez-vous aujourd'hui? Ceci est un exemple de synthèse vocale en français.",
                        None,
                        "French",
                        0.5,
                        0.5,
                        0.8,
                        0
                    ],
                    [
                        "你好，今天天气真不错。希望你有一个愉快的周末。",
                        None,
                        "Chinese",
                        0.5,
                        0.5,
                        0.8,
                        0
                    ],
                ] if USE_MULTILINGUAL else []),
                inputs=[text_input, audio_prompt, language, exaggeration, cfg_weight, temperature, seed],
                label="📚 Example Prompts"
            )

            # Tips
            gr.Markdown("""
                ### 💡 Tips for Best Results

                **Text Input:**
                - Use proper punctuation (. ! ? ,)
                - Keep sentences reasonably short
                - Avoid excessive special characters

                **Voice Cloning:**
                - 5-10 second samples work best
                - Use clear audio without background noise
                - Match the language of your voice sample to target language

                **Emotion Control:**
                - Start with exaggeration=0.5 (neutral)
                - Increase for more dramatic speech
                - Lower CFG weight for faster-paced speech

                **Speed:**
                - CPU: ~30-60s per generation
                - GPU: ~3-5s per generation
            """)

    # Footer
    gr.HTML("""
        <div class="footer">
            <p>
                Built with <a href="https://github.com/resemble-ai/chatterbox" target="_blank">Chatterbox</a> by
                <a href="https://resemble.ai" target="_blank">Resemble AI</a>
            </p>
            <p>All audio outputs include Perth watermarking for responsible AI.</p>
            <p>⚠️ Don't use this to create misleading or harmful content.</p>
        </div>
    """)

    # Connect button to function
    generate_btn.click(
        fn=generate_tts,
        inputs=[
            text_input,
            audio_prompt,
            language,
            exaggeration,
            cfg_weight,
            temperature,
            seed
        ],
        outputs=audio_output,
        api_name="generate"  # Enables programmatic access via /api/generate
    )

# Launch settings
if __name__ == "__main__":
    demo.queue(
        max_size=20,  # Maximum queue size
        default_concurrency_limit=2,  # Concurrent requests (adjust based on hardware)
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
