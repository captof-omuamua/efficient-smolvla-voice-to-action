"""Record from the laptop microphone and turn speech into text, all locally."""
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MODEL = "openai/whisper-large-v3-turbo"
RATE = 16000  # Whisper expects 16 kHz audio


def record_until_enter(path):
    """Record the microphone to a raw file until Enter is pressed. Returns the audio."""
    command = ["arecord", "-q", "-f", "S16_LE", "-r", str(RATE), "-c", "1", "-t", "raw", str(path)]
    recorder = subprocess.Popen(command)
    input("  Recording... press Enter when you finish speaking. ")
    recorder.terminate()
    recorder.wait()
    return load_raw(path)


def load_raw(path):
    """Read 16-bit raw audio into floats between -1 and 1."""
    return np.frombuffer(Path(path).read_bytes(), dtype=np.int16).astype(np.float32) / 32768.0


class Transcriber:
    def __init__(self, device="cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(MODEL)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()

    @torch.inference_mode()
    def __call__(self, audio):
        """Return (text, milliseconds taken)."""
        start = time.perf_counter()
        features = self.processor(audio, sampling_rate=RATE, return_tensors="pt").input_features
        ids = self.model.generate(
            features.to(self.device, dtype=torch.float16), language="en", task="transcribe", max_length=64
        )
        torch.cuda.synchronize()
        text = self.processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return text, (time.perf_counter() - start) * 1000
