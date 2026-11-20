import os
import json
import tempfile
import subprocess
import threading
import soundfile as sf
from typing import List, Dict, Optional, Tuple
from vosk import Model, KaldiRecognizer

import rag

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_AUDIO_DIR = os.path.join(BASE_DIR, "cache", "audio")
os.makedirs(CACHE_AUDIO_DIR, exist_ok=True)

# Lock for pyttsx3 COM access
tts_lock = threading.Lock()
vosk_lock = threading.Lock()
_vosk_model: Optional[Model] = None


def get_vosk_model() -> Optional[Model]:
    global _vosk_model
    with vosk_lock:
        if _vosk_model is None and os.path.exists(rag.VOSK_PATH):
            try:
                _vosk_model = Model(rag.VOSK_PATH)
            except Exception as e:
                print(f"[!] Error loading Vosk model: {e}")
        return _vosk_model


def get_available_voices() -> List[Dict[str, str]]:
    """Returns local TTS voices available via SAPI5."""
    voices_info = []
    try:
        import pyttsx3
        import pythoncom
        pythoncom.CoInitialize()
        eng = pyttsx3.init()
        for v in eng.getProperty("voices"):
            name = v.name
            gender = getattr(v, "gender", "Neutral") or "Neutral"
            clean_name = name.replace("Microsoft ", "").replace(" Desktop - English (United States)", "")
            voices_info.append({
                "id": v.id,
                "name": name,
                "label": f"{clean_name} ({gender})",
                "gender": str(gender)
            })
        pythoncom.CoUninitialize()
    except Exception as e:
        print(f"[!] Error querying TTS voices: {e}")
    return voices_info


def synthesize_speech(text: str, voice_id: Optional[str] = None, speed: float = 1.0) -> Tuple[bool, Optional[str], str]:
    """
    Synthesizes speech to a local WAV file completely offline.
    speed: 0.5 to 2.0 (normal 1.0 -> 150 words per minute)
    """
    if not text or not text.strip():
        return False, None, "Text cannot be empty."

    # Clean text of markdown, flowcharts, special symbols
    clean_text = text.replace("```", "").replace("[ ]", "").replace("-->", "to").replace("*", "")
    clean_text = clean_text[:2000].strip()  # Safe limit for single synthesis

    fd, out_path = tempfile.mkstemp(suffix=".wav", dir=CACHE_AUDIO_DIR)
    os.close(fd)

    with tts_lock:
        try:
            import pyttsx3
            import pythoncom
            pythoncom.CoInitialize()
            eng = pyttsx3.init()

            # Rate: default is ~150-175 WPM
            base_rate = 160
            calculated_rate = int(base_rate * max(0.5, min(2.0, speed)))
            eng.setProperty("rate", calculated_rate)

            # Voice selection
            if voice_id:
                try:
                    eng.setProperty("voice", voice_id)
                except Exception:
                    pass

            eng.save_to_file(clean_text, out_path)
            eng.runAndWait()
            pythoncom.CoUninitialize()

            if os.path.exists(out_path) and os.path.getsize(out_path) > 44:
                return True, out_path, "Synthesis successful."
            return False, None, "Synthesis failed (empty audio output)."
        except Exception as e:
            return False, None, f"TTS Error: {e}"


def transcribe_audio_file(file_path_or_bytes, suffix: str = ".wav") -> Tuple[bool, str, str]:
    """
    Transcribes any uploaded audio file offline using FFmpeg and Vosk.
    Returns (success, transcribed_text, message).
    """
    model = get_vosk_model()
    if model is None:
        return False, "", "Offline speech-to-text model not found on system."

    temp_input = None
    temp_wav = None

    try:
        # If raw bytes were passed, write to temporary file
        if isinstance(file_path_or_bytes, bytes):
            fd, temp_input = tempfile.mkstemp(suffix=suffix, dir=CACHE_AUDIO_DIR)
            with os.fdopen(fd, "wb") as f:
                f.write(file_path_or_bytes)
            input_path = temp_input
        else:
            input_path = str(file_path_or_bytes)

        if not os.path.exists(input_path):
            return False, "", "Audio input file not found."

        # Convert to standard 16kHz mono 16-bit PCM WAV using FFmpeg
        fd, temp_wav = tempfile.mkstemp(suffix=".wav", dir=CACHE_AUDIO_DIR)
        os.close(fd)

        ffmpeg_bin = rag.FFMPEG_PATH
        if not ffmpeg_bin or not os.path.exists(ffmpeg_bin):
            return False, "", "FFmpeg executable not found for audio conversion."

        cmd = [
            ffmpeg_bin, "-y",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            temp_wav
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode != 0 or not os.path.exists(temp_wav):
            return False, "", "Audio conversion to 16kHz WAV failed."

        # Read converted WAV
        data, sr = sf.read(temp_wav, dtype="int16")
        if len(data.shape) > 1:
            data = data.mean(axis=1).astype("int16")

        rec = KaldiRecognizer(model, 16000)
        results = []
        for i in range(0, len(data), 4000):
            chunk = data[i:i + 4000].tobytes()
            if rec.AcceptWaveform(chunk):
                part = json.loads(rec.Result()).get("text", "")
                if part:
                    results.append(part)

        final_part = json.loads(rec.FinalResult()).get("text", "")
        if final_part:
            results.append(final_part)

        transcription = " ".join(results).strip()
        return True, transcription, "Transcription successful."
    except Exception as e:
        return False, "", f"Transcription error: {e}"
    finally:
        if temp_input and os.path.exists(temp_input):
            try:
                os.remove(temp_input)
            except Exception:
                pass
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception:
                pass
