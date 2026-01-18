import whisper
import tempfile
import os

model = whisper.load_model("base")

def transcribe_whisper(audio_bytes) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            path = f.name

        result = model.transcribe(path)
        os.remove(path)
        return result["text"]
    except Exception:
        return ""

def extract_text_from_audio(audio_bytes) -> str:
    text = transcribe_whisper(audio_bytes)

    if len(text.strip()) > 10:
        return text

    return ""  # UI will ask user to retry or edit
