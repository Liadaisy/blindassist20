"""Voice synthesis engine for BlindAssist supporting English and Malayalam."""

import base64
import ctypes
import hashlib
import os
import queue
import subprocess
import threading
import time
import asyncio

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False


class Speaker:
    """Provides speech synthesis for safety warnings in English and Malayalam."""

    def __init__(self):
        self._speech_queue = queue.Queue(maxsize=3)
        self._worker_started = False
        self._queue_lock = threading.Lock()
        self._last_spoken_time = {}
        self._cache_dir = os.path.abspath("Data/audio_cache")
        os.makedirs(self._cache_dir, exist_ok=True)
        self._start_worker()
        self._pre_cache_common_alerts()

    def has_malayalam_voice(self):
        """Returns True if an online Malayalam TTS engine or Windows voice is available."""
        return EDGE_TTS_AVAILABLE or GTTS_AVAILABLE or self._has_windows_malayalam_voice()

    def _has_windows_malayalam_voice(self):
        try:
            command = (
                "Add-Type -AssemblyName System.Speech; "
                "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$voice = $speak.GetInstalledVoices() | "
                "Where-Object { $_.VoiceInfo.Culture.Name -like 'ml-*' } | "
                "Select-Object -First 1; "
                "if ($voice) { $voice.VoiceInfo.Name }"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                capture_output=True, text=True, timeout=5,
            )
            return bool(result.stdout.strip())
        except Exception:
            return False

    def _start_worker(self):
        with self._queue_lock:
            if not self._worker_started:
                self._worker_started = True
                threading.Thread(target=self._speech_worker, daemon=True).start()

    def _pre_cache_common_alerts(self):
        """Pre-generate common Malayalam safety alerts in the background."""
        common_phrases = [
            "ശ്രദ്ധിക്കുക. ഇടതുവശത്ത് ഓടയുണ്ട്. അരികിലേക്ക് പോകരുത്.",
            "ശ്രദ്ധിക്കുക. വലതുവശത്ത് ഓടയുണ്ട്. അരികിലേക്ക് പോകരുത്.",
            "മുന്നറിയിപ്പ്. മുന്നിൽ റോഡിൽ കുഴിയുണ്ട്. ശ്രദ്ധിച്ചു നടക്കുക.",
            "മുന്നറിയിപ്പ്. വലതുവശത്ത് റോഡിൽ കുഴിയുണ്ട്. ശ്രദ്ധിച്ചു നടക്കുക.",
            "മുന്നറിയിപ്പ്. ഇടതുവശത്ത് റോഡിൽ കുഴിയുണ്ട്. ശ്രദ്ധിച്ചു നടക്കുക.",
            "ശ്രദ്ധിക്കുക. മുന്നിൽ വെള്ളക്കെട്ടുണ്ട്.",
            "ശ്രദ്ധിക്കുക. ഇടതുവശത്ത് വെള്ളക്കെട്ടുണ്ട്.",
            "ശ്രദ്ധിക്കുക. വലതുവശത്ത് വെള്ളക്കെട്ടുണ്ട്.",
            "മുന്നറിയിപ്പ്. വാഹനം മുന്നിൽ അതിവേഗം അടുത്തുവരുന്നു.",
            "മുന്നറിയിപ്പ്. കാർ മുന്നിൽ അതിവേഗം അടുത്തുവരുന്നു.",
            "മുന്നറിയിപ്പ്. ബസ് മുന്നിൽ അതിവേഗം അടുത്തുവരുന്നു.",
            "മുന്നറിയിപ്പ്. മോട്ടോർസൈക്കിൾ മുന്നിൽ അതിവേഗം അടുത്തുവരുന്നു.",
            "മുന്നറിയിപ്പ്. പടികൾ മുന്നിൽ വളരെ അടുത്തുണ്ട്.",
        ]

        def _caching_task():
            for phrase in common_phrases:
                phrase_hash = hashlib.md5(phrase.encode("utf-8")).hexdigest()
                cache_path = os.path.join(self._cache_dir, f"{phrase_hash}.mp3")
                if not os.path.exists(cache_path):
                    try:
                        if EDGE_TTS_AVAILABLE:
                            asyncio.run(self._generate_edge_tts(phrase, cache_path))
                        elif GTTS_AVAILABLE:
                            gTTS(phrase, lang="ml").save(cache_path)
                    except Exception as exc:
                        print(f"[SPEAKER] Pre-cache failed: {exc}")

        threading.Thread(target=_caching_task, daemon=True).start()

    async def _generate_edge_tts(self, text, cache_path):
        communicate = edge_tts.Communicate(text, voice="ml-IN-MidhunNeural")
        await communicate.save(cache_path)

    def speak(self, text):
        """Speak standard English text using Windows speech synthesizer."""
        print("Speaking:", text)
        try:
            self._speak_with_windows(text, malayalam=False)
        except Exception as exc:
            print(f"[SPEAKER ERROR] Unable to speak English: {exc}")

    def speak_malayalam(self, text):
        """Queue Malayalam text for speech output without blocking the video loop."""
        text = str(text).strip()
        if not text:
            return
        now = time.monotonic()
        last_time = self._last_spoken_time.get(text, 0.0)
        if now - last_time < 2.5:
            return
        self._last_spoken_time[text] = now
        self._start_worker()
        try:
            self._speech_queue.put_nowait(text)
        except queue.Full:
            try:
                self._speech_queue.get_nowait()
                self._speech_queue.task_done()
            except queue.Empty:
                pass
            self._speech_queue.put_nowait(text)

    def _speech_worker(self):
        while True:
            text = self._speech_queue.get()
            try:
                self._speak_malayalam_impl(text)
            except Exception as exc:
                print(f"[SPEAKER ERROR] Unable to speak Malayalam: {exc}")
            finally:
                self._speech_queue.task_done()

    def _speak_malayalam_impl(self, text):
        phrase_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        cache_path = os.path.join(self._cache_dir, f"{phrase_hash}.mp3")

        if not os.path.exists(cache_path) or os.path.getsize(cache_path) == 0:
            generated = False
            if EDGE_TTS_AVAILABLE:
                try:
                    asyncio.run(self._generate_edge_tts(text, cache_path))
                    generated = os.path.exists(cache_path) and os.path.getsize(cache_path) > 0
                except Exception as exc:
                    print(f"[SPEAKER WARNING] Edge TTS generation failed: {exc}")
            if not generated and GTTS_AVAILABLE:
                try:
                    gTTS(text, lang="ml").save(cache_path)
                    generated = os.path.exists(cache_path) and os.path.getsize(cache_path) > 0
                except Exception as exc:
                    print(f"[SPEAKER WARNING] gTTS generation failed: {exc}")

        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            if self._play_audio_file(cache_path):
                return

        if self._has_windows_malayalam_voice():
            self._speak_with_windows(text, malayalam=True)
        else:
            print("[SPEAKER ERROR] Malayalam audio unavailable. Check internet and audio output.")

    @staticmethod
    def _play_audio_file(audio_path):
        """Play MP3 file using winmm MCI, then try playsound3."""
        try:
            alias = f"spk_{os.path.basename(audio_path).split('.')[0][:8]}"
            winmm = ctypes.windll.winmm
            winmm.mciSendStringW(f"close {alias}", None, 0, None)
            r_open = winmm.mciSendStringW(
                f'open "{audio_path}" type mpegvideo alias {alias}', None, 0, None
            )
            if r_open == 0:
                r_play = winmm.mciSendStringW(f"play {alias} wait", None, 0, None)
                winmm.mciSendStringW(f"close {alias}", None, 0, None)
                if r_play == 0:
                    return True
        except Exception as exc:
            print(f"[SPEAKER] winmm playback failed: {exc}")
        try:
            from playsound3 import playsound
            playsound(audio_path, block=True)
            return True
        except Exception as exc:
            print(f"[SPEAKER] playsound3 playback failed: {exc}")
            return False

    @staticmethod
    def _speak_with_windows(text, malayalam=False):
        voice_selection = ""
        if malayalam:
            voice_selection = """
$voice = $speak.GetInstalledVoices() |
    Where-Object { $_.VoiceInfo.Culture.Name -like 'ml-*' } |
    Select-Object -First 1
if (-not $voice) { throw "No Malayalam Windows speech voice is installed." }
$speak.SelectVoice($voice.VoiceInfo.Name)
"""
        command = f"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speak.Rate = 0
{voice_selection}
$encodedText = [Console]::In.ReadToEnd()
$text = [System.Text.Encoding]::UTF8.GetString(
    [System.Convert]::FromBase64String($encodedText.Trim())
)
$speak.Speak($text)
"""
        encoded_text = base64.b64encode(text.encode("utf-8")).decode("ascii")
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            input=encoded_text, capture_output=True, text=True, check=True,
        )
