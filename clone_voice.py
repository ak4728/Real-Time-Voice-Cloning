"""
Non-interactive voice cloning script.
Usage: python clone_voice.py <audio_file> <"text to synthesize"> [output.wav] [--fast]
  --fast   Use batched vocoder (faster but lower quality). Default is high-quality unbatched.
"""
import os
import sys
from pathlib import Path

# Ensure system PATH is loaded so ffmpeg (needed for MP3/M4A) is found
os.environ["PATH"] = (
    os.environ.get("PATH", "") + os.pathsep +
    os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links")
)
import numpy as np
import soundfile as sf

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
from utils.default_models import ensure_default_models

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

    print(f"Synthesizing: {text!r}")
    mel = synthesizer.synthesize_spectrograms([text], [embed])[0]

    print("Generating waveform...")
    if fast:
        # Batched — faster, slightly lower quality
        wav_out = vocoder.infer_waveform(mel, target=8000, overlap=800)
    else:
        # Unbatched — slower but noticeably clearer enunciation
        wav_out = vocoder.infer_waveform(mel, batched=False)

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
