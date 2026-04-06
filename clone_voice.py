"""
Non-interactive voice cloning script.
Usage: python clone_voice.py <audio_file> <"text to synthesize"> [output.wav] [--fast]
  --fast   Use batched vocoder (faster but lower quality). Default is high-quality unbatched.
"""
import os
import sys
import warnings
from pathlib import Path

# Ensure system PATH is loaded so ffmpeg (needed for MP3/M4A) is found
os.environ["PATH"] = (
    os.environ.get("PATH", "") + os.pathsep +
    os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links")
)
import numpy as np
import soundfile as sf

# Suppress expected MP3 fallback warning (soundfile doesn't support MP3; audioread/ffmpeg is used instead)
warnings.filterwarnings("ignore", message="PySoundFile failed", category=UserWarning)

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
from utils.default_models import ensure_default_models

def split_sentences(text: str):
    """Split text into sentences so Tacotron doesn't lose alignment on long inputs."""
    import re
    # Split on sentence-ending punctuation, keeping the delimiter
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    # Drop empty strings
    return [p.strip() for p in parts if p.strip()]


def clone(audio_path: str, text: str, out_path: str = "cloned_output.wav", fast: bool = False):
    ensure_default_models(Path("saved_models"))

    print("Loading models...")
    encoder.load_model(Path("saved_models/default/encoder.pt"))
    synthesizer = Synthesizer(Path("saved_models/default/synthesizer.pt"))
    vocoder.load_model(Path("saved_models/default/vocoder.pt"))

    print(f"Processing reference audio: {audio_path}")
    wav = encoder.preprocess_wav(Path(audio_path))
    embed = encoder.embed_utterance(wav)
    print("Speaker embedding computed.")

    sentences = split_sentences(text)
    print(f"Synthesizing {len(sentences)} sentence(s)...")

    silence = np.zeros(int(synthesizer.sample_rate * 0.15))  # 150ms gap between sentences
    wav_segments = []

    for i, sentence in enumerate(sentences, 1):
        print(f"  [{i}/{len(sentences)}] {sentence!r}")
        mel = synthesizer.synthesize_spectrograms([sentence], [embed])[0]
        if fast:
            seg = vocoder.infer_waveform(mel, target=8000, overlap=800)
        else:
            seg = vocoder.infer_waveform(mel, batched=False)
        wav_segments.append(seg)
        if i < len(sentences):
            wav_segments.append(silence)

    wav_out = np.concatenate(wav_segments)
    wav_out = np.pad(wav_out, (0, synthesizer.sample_rate), mode="constant")

    sf.write(out_path, wav_out.astype(np.float32), synthesizer.sample_rate)
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python clone_voice.py <audio_file> <text> [output.wav] [--fast]")
        sys.exit(1)
    fast_mode = "--fast" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--fast"]
    out = args[2] if len(args) > 2 else "cloned_output.wav"
    clone(args[0], args[1], out, fast=fast_mode)
