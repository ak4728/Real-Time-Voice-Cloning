"""
Non-interactive voice cloning script.
Usage: python clone_voice.py <audio_file> <"text to synthesize"> [output.wav]
"""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
from utils.default_models import ensure_default_models

def clone(audio_path: str, text: str, out_path: str = "cloned_output.wav"):
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
    wav_out = vocoder.infer_waveform(mel)
    wav_out = np.pad(wav_out, (0, synthesizer.sample_rate), mode="constant")

    sf.write(out_path, wav_out.astype(np.float32), synthesizer.sample_rate)
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python clone_voice.py <audio_file> <text> [output.wav]")
        sys.exit(1)
    out = sys.argv[3] if len(sys.argv) > 3 else "cloned_output.wav"
    clone(sys.argv[1], sys.argv[2], out)
