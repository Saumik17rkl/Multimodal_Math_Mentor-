import tempfile
import subprocess
import os

def extract_text_from_audio(audio_bytes) -> str:
    """
    Robust audio-to-text with proper decoding.
    """

    # Save raw input
    raw_file = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    raw_file.write(audio_bytes)
    raw_file.close()

    wav_path = raw_file.name.replace(".webm", ".wav")

    # ---------- CONVERT USING FFMPEG ----------
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", raw_file.name,
                "-ac", "1",
                "-ar", "16000",
                wav_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except Exception:
        return ""

    # ---------- WHISPER ----------
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(wav_path)
        text = result.get("text", "").strip()
        if len(text) > 10:
            return text
    except Exception:
        pass

    # ---------- VOSK ----------
    try:
        import wave, json
        from vosk import Model, KaldiRecognizer

        model = Model("vosk-model-small-en-us-0.15")
        wf = wave.open(wav_path, "rb")

        rec = KaldiRecognizer(model, wf.getframerate())
        text_parts = []

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                text_parts.append(json.loads(rec.Result()).get("text", ""))

        text_parts.append(json.loads(rec.FinalResult()).get("text", ""))
        text = " ".join(text_parts).strip()
        if len(text) > 10:
            return text
    except Exception:
        pass

    return ""
