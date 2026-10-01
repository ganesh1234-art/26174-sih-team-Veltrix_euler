"""Offline voice, translation, Bluetooth discovery, and RAM-aware Ollama access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Callable
import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


DEFAULT_ALERTS = (
    "Recording started",
    "Recording stopped",
    "Step passed",
    "Experiment complete",
    "Voice chat is ready",
)


class TranslationCache:
    """Disk-backed translation cache; Argos is invoked only for a cache miss."""

    def __init__(self, cache_path: str | Path = ".sih_runtime/phrase_cache.json") -> None:
        self.path = Path(cache_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        try:
            self._values: dict[str, str] = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            self._values = {}

    @staticmethod
    def _key(text: str, source: str, target: str) -> str:
        return f"{source.lower()}::{target.lower()}::{text}"

    def translate(self, text: str, source: str = "en", target: str = "en") -> str:
        if not text or source.lower() == target.lower():
            return text
        key = self._key(text, source, target)
        with self._lock:
            cached = self._values.get(key)
        if cached is not None:
            return cached
        translated = text
        try:
            import argostranslate.translate as argos_translate

            translated = argos_translate.translate(text, source, target)
        except ImportError:
            logger.warning("argos-translate is unavailable; speaking source language")
        except Exception as exc:  # package missing, unsupported pair, or malformed model
            logger.warning("offline translation unavailable (%s); speaking source language", exc)
        with self._lock:
            self._values[key] = translated
            self.path.write_text(json.dumps(self._values, ensure_ascii=False, indent=2), encoding="utf-8")
        return translated

    def warm(self, phrases: tuple[str, ...] = DEFAULT_ALERTS, source: str = "en", target: str = "en") -> None:
        for phrase in phrases:
            self.translate(phrase, source, target)


class OfflineSpeaker:
    """Non-blocking pyttsx3 speaker with phrase translation performed before TTS.

    Three properties matter for a PySide6 application:

    * ``say()`` only enqueues and returns immediately, so a GUI thread never
      waits for an utterance.
    * pyttsx3 is a COM/SAPI wrapper, and SAPI5 refuses to work on a thread whose
      apartment is not initialised. ``pythoncom.CoInitialize()`` is therefore
      called at the start of the dedicated speech thread on Windows; without it
      the first utterance after a thread switch fails silently.
    * The engine is created *inside* that same thread, because COM objects are
      bound to the apartment that created them.
    """

    def __init__(self, translation_cache: TranslationCache | None = None) -> None:
        self.cache = translation_cache or TranslationCache()
        self._queue: Queue[tuple[str, str, str]] = Queue(maxsize=16)
        self._stop = Event()
        self._thread = Thread(target=self._run, name="offline-tts", daemon=True)
        self._thread.start()
        self.ready = Event()
        self.last_error: str | None = None
        self.spoken_count = 0
        self.failed_count = 0
        self.dropped_count = 0
        #: Engine created once on the speech thread and reused. Building a new
        #: pyttsx3 engine per utterance leaves the SAPI5 COM object in a bad
        #: state and is the usual cause of "speaks once, then never again".
        self._engine: object | None = None

    def say(self, text: str, source_language: str = "en", target_language: str = "en") -> bool:
        """Queue an utterance. Returns immediately and never blocks the caller."""

        if not str(text).strip():
            return False
        try:
            self._queue.put_nowait((str(text), source_language, target_language))
            return True
        except Exception:
            self.dropped_count += 1
            logger.warning("Speech queue full; dropped %r", text)
            return False

    @property
    def status(self) -> str:
        """Short human-readable state, surfaced in the UI status bar."""

        if not self._thread.is_alive():
            return "TTS thread is not running"
        if self.last_error:
            return f"TTS problem: {self.last_error}"
        if not self._engine:
            return "TTS ready, waiting for the first announcement"
        return f"TTS ready ({self.spoken_count} spoken, {self.failed_count} failed)"

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    @staticmethod
    def _init_com() -> bool:
        """Initialise COM for the calling thread. Returns True when initialised."""

        if os.name != "nt":
            return False
        try:
            import pythoncom

            pythoncom.CoInitialize()
            return True
        except Exception as exc:  # pywin32 missing: fall back to plain init()
            logger.warning("pythoncom.CoInitialize unavailable: %s", exc)
            return False

    @staticmethod
    def _release_com(initialised: bool) -> None:
        if not initialised:
            return
        try:
            import pythoncom

            pythoncom.CoUninitialize()
        except Exception as exc:
            logger.debug("CoUninitialize failed: %s", exc)

    def _run(self) -> None:
        com_initialised = self._init_com()
        try:
            self.ready.set()
            while not self._stop.is_set():
                try:
                    text, source, target = self._queue.get(timeout=0.25)
                except Empty:
                    continue
                # Nothing in the loop may be allowed to kill the speech thread:
                # once it dies every later alert is silently lost.
                try:
                    spoken = f"System notice: {self.cache.translate(text, source, target)}"
                    self._speak_one(spoken)
                except Exception as exc:
                    self.failed_count += 1
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    logger.error("Speech worker error: %s", self.last_error)
        finally:
            self._dispose_engine()
            self._release_com(com_initialised)

    def _engine_or_none(self) -> object | None:
        """Return the reused engine, creating it once on this COM-initialised thread."""

        if self._engine is not None:
            return self._engine
        self._engine = self._new_engine()
        return self._engine

    def _dispose_engine(self) -> None:
        engine, self._engine = self._engine, None
        if engine is None:
            return
        try:
            engine.stop()
        except Exception:
            pass

    def _speak_one(self, text: str) -> None:
        engine = None
        try:
            engine = self._engine_or_none()
            engine.say(text)
            engine.runAndWait()
            self.last_error = None
            self.spoken_count += 1
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.failed_count += 1
            logger.error("TTS failed: %s", self.last_error)
            # A wedged SAPI5 engine must not poison every later announcement.
            self._dispose_engine()
            self._fallback_wave(engine, text)

    @staticmethod
    def _new_engine() -> object:
        """Create and configure a pyttsx3 engine on the current (COM-ready) thread."""

        import pyttsx3

        engine = pyttsx3.init()
        voices = engine.getProperty("voices") or []
        preferred = ("david", "mark", "george", "jarvis")
        voice = next(
            (item for item in voices if any(name in str(getattr(item, "name", "")).casefold() for name in preferred)),
            None,
        )
        if voice is not None:
            engine.setProperty("voice", voice.id)
        engine.setProperty("rate", 170)
        engine.setProperty("volume", 1.0)
        return engine

    @staticmethod
    def _fallback_wave(engine: object | None, text: str) -> None:
        path = Path(tempfile.gettempdir()) / "sih_jarvis_alert.wav"
        try:
            if engine is not None:
                engine.save_to_file(text, str(path))
                engine.runAndWait()
            else:
                escaped = text.replace("'", "''")
                command = (
                    "Add-Type -AssemblyName System.Speech; "
                    "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$s.SetOutputToWaveFile('{str(path).replace(chr(39), chr(39) * 2)}'); "
                    f"$s.Speak('{escaped}'); $s.Dispose()"
                )
                subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command], check=True, timeout=15)
            import wave
            import numpy as np
            import sounddevice as sd

            with wave.open(str(path), "rb") as handle:
                frames = handle.readframes(handle.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16)
                channels = handle.getnchannels()
                if channels > 1:
                    audio = audio.reshape(-1, channels)
                sd.play(audio, handle.getframerate(), blocking=True)
        except Exception as exc:
            logger.warning("WAV audio fallback failed: %s", exc)
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


class BluetoothAudioManager:
    """BLE discovery/connection helper.

    A2DP audio routing itself is owned by the OS Bluetooth stack.  Pair the
    speaker in the OS and select it as the default output; pyttsx3 will then
    broadcast through it.  This helper automates BLE peripheral discovery and
    retains a connection for devices that expose a BLE control service.
    """

    def __init__(self) -> None:
        self.client = None
        self.connected_address: str | None = None

    @staticmethod
    def output_devices() -> list[dict[str, object]]:
        """Return current PortAudio render devices, if sounddevice is available."""

        try:
            import sounddevice as sd

            devices = sd.query_devices()
            return [
                {
                    "index": index,
                    "name": str(device["name"]),
                    "hostapi": int(device["hostapi"]),
                    "max_output_channels": int(device["max_output_channels"]),
                }
                for index, device in enumerate(devices)
                if int(device["max_output_channels"]) > 0
            ]
        except Exception as exc:
            logger.warning("Audio output enumeration failed: %s", exc)
            return []

    @classmethod
    def resolve_output_device(cls, device: int | str) -> dict[str, object] | None:
        """Resolve a current output device; names survive reconnects better than indices."""

        devices = cls.output_devices()
        if isinstance(device, int):
            return next((item for item in devices if item["index"] == device), None)
        query = device.casefold().strip()
        return next((item for item in devices if query in str(item["name"]).casefold()), None)

    def discover(self, timeout_seconds: float = 5.0) -> list[dict[str, str]]:
        """Compatibility helper for non-async callers without leaking coroutines."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.discover_async(timeout_seconds))
        # A running event loop is common in GUI frameworks. Those callers
        # should directly await ``discover_async`` instead.
        logger.warning("Use discover_async from an active event loop")
        return []

    async def discover_async(self, timeout_seconds: float = 5.0) -> list[dict[str, str]]:
        try:
            from bleak import BleakScanner

            devices = await BleakScanner.discover(timeout=timeout_seconds)
            return [{"name": device.name or "Unnamed BLE device", "address": device.address} for device in devices]
        except Exception as exc:
            logger.warning("BLE discovery failed: %s", exc)
            return []

    async def connect(self, address: str) -> bool:
        await self.disconnect()
        try:
            from bleak import BleakClient

            client = BleakClient(address, disconnected_callback=self._on_disconnected)
            self.client = client
            await client.connect()
            self.connected_address = address
            return bool(client.is_connected)
        except Exception as exc:
            logger.warning("BLE connection failed: %s", exc)
            self.client = None
            self.connected_address = None
            return False

    async def disconnect(self) -> None:
        if self.client is not None:
            try:
                await self.client.disconnect()
            except Exception as exc:
                logger.debug("BLE disconnect failed: %s", exc)
        self.client = None
        self.connected_address = None

    def _on_disconnected(self, client: object) -> None:
        if client is self.client:
            self.connected_address = None
            logger.warning("BLE control connection dropped")


@dataclass(slots=True)
class OllamaRouter:
    model: str = "llama3.1:8b-instruct-q4_K_M"
    base_url: str = "http://127.0.0.1:11434"
    keep_alive: str = "120s"
    timeout_seconds: float = 90.0
    PLANNER_SCHEMA = {
        "type": "object",
        "properties": {
            "classes": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "sequence": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "integer", "minimum": 1},
                        "target": {"type": "string"},
                        "action": {"type": "string"},
                        "parent": {"type": "string"},
                    },
                    "required": ["step", "target", "action"],
                },
            },
        },
        "required": ["classes", "sequence"],
    }

    def available_models(self) -> list[str]:
        """Model names currently installed in the local Ollama service."""

        try:
            with urllib.request.urlopen(f"{self.base_url.rstrip('/')}/api/tags", timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not list Ollama models: %s", exc)
            return []
        names = [str(item.get("name", "")) for item in payload.get("models", [])]
        return [name for name in names if name]

    def resolve_model(self) -> str:
        """Return a model tag that actually exists locally.

        The configured default is a preferred tag, not a requirement. Ollama
        answers 404 for a tag it never pulled, which used to make every voice
        chat query fail even though a perfectly good model was installed.
        """

        installed = self.available_models()
        if not installed:
            return self.model
        if self.model in installed:
            return self.model
        family = self.model.split(":")[0]
        for name in installed:
            if name.split(":")[0] == family:
                # Same family, different tag: the configured one was never pulled.
                logger.info("Using installed Ollama model %s instead of %s", name, self.model)
                return name
        logger.warning(
            "Configured Ollama model %s is not installed; using %s",
            self.model,
            installed[0],
        )
        return installed[0]

    def _post(self, endpoint: str, payload: dict[str, object], timeout: float | None = None) -> dict[str, object]:
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

    def ask(self, prompt: str) -> str:
        """Query the local LLM, keeping it resident for the keep-alive window.

        A first request can fail while Ollama is still loading the weights, so
        one retry is attempted before giving up.
        """

        payload = {
            "model": self.resolve_model(),
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
        }
        try:
            result = self._post("/api/generate", payload)
        except RuntimeError as exc:
            logger.info("Ollama retrying after: %s", exc)
            result = self._post("/api/generate", payload)
        return str(result.get("response", ""))

    def plan_experiment(self, experiment_text: str) -> dict[str, object]:
        """Convert experiment text to validated JSON data, never executable code."""

        source = experiment_text.strip()
        if not source:
            raise ValueError("Experiment text is empty.")
        if len(source) > 12_000:
            raise ValueError("Experiment text exceeds the offline planner limit of 12000 characters.")
        prompt = (
            "Parse the experiment below into the requested JSON schema. "
            "Return only experiment classes and ordered observation actions. "
            "Never return Python, shell commands, imports, or executable expressions.\n\n"
            f"Experiment:\n{source}"
        )
        try:
            result = self._post(
                "/api/generate",
                {
                    "model": self.resolve_model(),
                    "prompt": prompt,
                    "system": "You are a deterministic offline experiment-data parser.",
                    "stream": False,
                    "format": self.PLANNER_SCHEMA,
                    "keep_alive": "30s",
                    "options": {"temperature": 0, "num_ctx": 2048},
                },
            )
            raw_response = result.get("response")
            if not isinstance(raw_response, str):
                raise RuntimeError("Ollama planner returned no JSON response.")
            planned = json.loads(raw_response)
            if not isinstance(planned, dict):
                raise ValueError("Ollama planner response must be a JSON object.")
            return planned
        finally:
            try:
                self.unload()
            except RuntimeError as exc:
                logger.warning("Ollama planner unload failed: %s", exc)

    def unload(self) -> None:
        """Immediately flush model weights when Voice Chat is switched off."""

        self._post(
            "/api/generate",
            {"model": self.resolve_model(), "prompt": "", "stream": False, "keep_alive": 0},
            timeout=10,
        )


class VoskWakeDaemon:
    """Vosk + sounddevice daemon that sends only wake-word queries to the LLM."""

    def __init__(
        self,
        model_path: str | Path,
        wake_word: str = "assistant",
        on_query: Callable[[str], None] | None = None,
        on_awake: Callable[[], None] | None = None,
        sample_rate: int = 16_000,
        query_timeout_seconds: float = 12.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.wake_word = wake_word.lower().strip()
        self.on_query = on_query
        self.on_awake = on_awake
        self.sample_rate = sample_rate
        self.query_timeout_seconds = query_timeout_seconds
        self._audio: Queue[bytes] = Queue(maxsize=24)
        self._stop = Event()
        self._ready = Event()
        self._thread: Thread | None = None
        self.last_error: str | None = None
        self.transcript_count = 0
        self.wake_count = 0
        self.answered_count = 0
        self.input_device: str | None = None
        self._idle_recognizer: object | None = None
        self._open_recognizer: object | None = None
        self._awaiting_query = False
        self._awaiting_since = 0.0
        self._partial_wake_fired = False

    @property
    def awaiting_question(self) -> bool:
        return self._awaiting_query

    def _wake_word_heard(self, transcript: str) -> bool:
        """Tolerant wake-word test.

        A small ASR model truncates or mangles a single spoken word, so an exact
        match alone misses most wake attempts. A clear prefix counts too, which
        covers "assis" and "assistan" style transcripts.
        """

        if not self.wake_word:
            return False
        if self.wake_word in transcript:
            return True
        # Five characters is distinctive enough for a wake word and tolerant of
        # a small ASR model truncating the final syllable.
        stem = self.wake_word[:5] if len(self.wake_word) > 5 else self.wake_word
        return bool(stem) and stem in transcript

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> bool:
        """Start the daemon and confirm the microphone stream really opened.

        Returning ``True`` as soon as the thread launched was the reason a dead
        wake-word listener still reported "enabled": the model load and the
        audio stream both happen inside the thread, so a failure there was only
        visible in ``last_error``.
        """

        if self.running:
            return True
        if not self.model_path.is_dir():
            self.last_error = f"Vosk model folder not found: {self.model_path}"
            return False
        self.last_error = None
        self._stop.clear()
        self._thread = Thread(target=self._run, name="vosk-wake-word", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if not self.running:
                break
            if self._ready.is_set():
                return True
            time.sleep(0.02)
        if not self._ready.is_set():
            self.last_error = self.last_error or "Vosk daemon did not start the microphone"
            return False
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._thread = None
        self._ready.clear()

    def _callback(self, indata: bytes, frames: int, timing: object, status: object) -> None:
        if status:
            logger.debug("audio status: %s", status)
        try:
            self._audio.put_nowait(bytes(indata))
        except Exception:
            pass  # Drop stale audio rather than spending RAM on an unbounded queue.

    def _run(self) -> None:
        stream = None
        try:
            import sounddevice as sd
            from vosk import KaldiRecognizer, Model, SetLogLevel

            SetLogLevel(-1)
            info = sd.query_devices(kind="input")
            logger.info("Wake-word listener using input device: %s", info["name"])
            self.input_device = str(info["name"])
            model = Model(str(self.model_path))
            # The wake-word grammar is what makes this small model usable:
            # measured on real speech it recognises the wake word 4/4, while the
            # unrestricted recogniser turns "assistant" into "a strange" and the
            # tolerant matcher cannot see it. Partial results are read as well
            # because a small model rarely declares end-of-speech for one word.
            self._idle_recognizer = KaldiRecognizer(
                model, self.sample_rate, json.dumps([self.wake_word, "[unk]"])
            )
            self._open_recognizer = None
            stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=4000,
                dtype="int16",
                channels=1,
                callback=self._callback,
            )
            stream.start()
            # Only now is the microphone genuinely listening.
            self._ready.set()
            while not self._stop.is_set():
                try:
                    data = self._audio.get(timeout=0.25)
                except Empty:
                    continue
                recognizer = self._recognizer()
                self._consume(recognizer, data)
        except Exception as exc:
            self.last_error = str(exc)
            logger.warning("Vosk daemon stopped: %s", exc)
        finally:
            self._ready.clear()
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

    def _consume(self, recognizer: object, data: bytes) -> None:
        """Feed one audio block and read the wake word wherever it appears.

        Only reading ``Result()`` was the reason a single short wake word was
        never recognised: a small model such as vosk-model-small-en-in often
        never declares end-of-speech for one word, so the final result never
        arrives. ``PartialResult()`` is emitted continuously, so the wake word
        is taken from there as soon as it is spoken.
        """

        try:
            final = bool(recognizer.AcceptWaveform(data))  # type: ignore[attr-defined]
        except Exception as exc:
            logger.debug("AcceptWaveform failed: %s", exc)
            return
        if final:
            self._partial_wake_fired = False
            try:
                text = str(json.loads(recognizer.Result()).get("text", "")).lower().strip()  # type: ignore[attr-defined]
            except (json.JSONDecodeError, AttributeError):
                return
            if text:
                self.transcript_count += 1
            self._handle_transcript(text)
            return
        if self._partial_wake_fired or self._awaiting_query:
            return
        try:
            partial = str(
                json.loads(recognizer.PartialResult()).get("partial", "")  # type: ignore[attr-defined]
            ).lower()
        except (json.JSONDecodeError, AttributeError):
            return
        if partial and self._wake_word_heard(partial):
            self._partial_wake_fired = True
            self._handle_transcript(partial)

    def _recognizer(self) -> object:
        """Wake-word grammar while idle, full vocabulary while expecting a question."""

        if self._awaiting_query:
            if self._open_recognizer is None:
                try:
                    from vosk import KaldiRecognizer, Model

                    self._open_recognizer = KaldiRecognizer(
                        Model(str(self.model_path)), self.sample_rate
                    )
                except Exception as exc:
                    logger.warning("Could not open the question recogniser: %s", exc)
                    return self._idle_recognizer
            return self._open_recognizer
        if self._open_recognizer is not None:
            self._open_recognizer = None
        return self._idle_recognizer

    def _handle_transcript(self, transcript: str) -> None:
        """Route one recognized phrase through the wake/ask state machine.

        Two ways in: the wake word is used on its own, in which case the listener
        announces a prompt and then waits for the *next* phrase to use as the
        question; or the wake word shares the phrase with the question, in which
        case it is answered straight away.
        """

        if not transcript:
            return
        if self._awaiting_query:
            if time.monotonic() - self._awaiting_since > self.query_timeout_seconds:
                self._awaiting_query = False
            else:
                self._awaiting_query = False
                self.answered_count += 1
                if self.on_query:
                    Thread(target=self.on_query, args=(transcript,), name="wake-query", daemon=True).start()
                return
        if not self._wake_word_heard(transcript):
            return
        position = transcript.find(self.wake_word)
        trailing = (
            transcript[position + len(self.wake_word) :].strip(" ,.?!") if position >= 0 else ""
        )
        if trailing:
            self.answered_count += 1
            if self.on_query:
                Thread(target=self.on_query, args=(trailing,), name="wake-query", daemon=True).start()
            return
        # Wake word alone: prompt, then listen for the question itself.
        self._awaiting_query = True
        self._awaiting_since = time.monotonic()
        self.wake_count += 1
        if self.on_awake:
            Thread(target=self.on_awake, name="wake-ack", daemon=True).start()


class AudioBrain:
    """Single facade owned by Streamlit session state."""

    def __init__(
        self,
        cache_path: str | Path = ".sih_runtime/phrase_cache.json",
        ollama_model: str = "llama3.1:8b",
    ) -> None:
        self.translations = TranslationCache(cache_path)
        self.speaker = OfflineSpeaker(self.translations)
        self.ollama = OllamaRouter(model=ollama_model)
        self.voice_chat_enabled = False
        self.wake_daemon: VoskWakeDaemon | None = None
        self.last_reply = ""

    def warm_alerts(self, language: str = "en") -> None:
        self.translations.warm(target=language)

    def announce(self, phrase: str, language: str = "en") -> None:
        """Speak a phrase without blocking the caller (queued on the TTS thread)."""

        self.speaker.say(phrase, target_language=language)

    def set_voice_chat(
        self, enabled: bool, vosk_model_path: str | Path | None = None, wake_word: str = "assistant"
    ) -> str:
        """Enable or disable wake-word voice chat. Always returns a report.

        The microphone and the model are opened on the daemon thread and can
        fail there, so every step is checked and the reason is returned instead
        of leaving the caller with a silent no-op.
        """

        self.voice_chat_enabled = enabled
        if not enabled:
            if self.wake_daemon:
                self.wake_daemon.stop()
            try:
                self.ollama.unload()
                return "Voice chat disabled; Ollama unload request sent."
            except RuntimeError as exc:
                return f"Voice chat disabled; Ollama was not reachable ({exc})."
        if not vosk_model_path:
            return "Voice chat needs a Vosk model folder in Settings before it can listen."
        if not Path(vosk_model_path).is_dir():
            message = f"Vosk model folder not found: {vosk_model_path}"
            self.voice_chat_enabled = False
            return message
        if not wake_word.strip():
            wake_word = "assistant"
        self.wake_daemon = VoskWakeDaemon(
            vosk_model_path,
            wake_word,
            self._respond_to_wake_query,
            on_awake=self._acknowledge_wake,
        )
        if not self.wake_daemon.start():
            reason = self.wake_daemon.last_error or "unknown error"
            self.voice_chat_enabled = False
            self.wake_daemon = None
            return f"Voice chat could not start: {reason}"
        return (
            f"Voice chat enabled. Listening for '{self.wake_daemon.wake_word}'; "
            f"Ollama stays unloaded until the wake word is recognised."
        )

    def voice_chat_status(self) -> str:
        """One-line report for the status bar, safe to call at any time."""

        if not self.voice_chat_enabled:
            return "Voice chat off"
        daemon = self.wake_daemon
        if daemon is None or not daemon.running:
            reason = (daemon.last_error if daemon else None) or "not running"
            return f"Voice chat on but the listener is down: {reason}"
        return f"Voice chat on, listening for '{daemon.wake_word}' ({daemon.transcript_count} phrases heard)"

    def wake_chat_healthy(self) -> bool:
        """True when voice chat is enabled and its listener is actually alive."""

        if not self.voice_chat_enabled:
            return True
        return bool(self.wake_daemon and self.wake_daemon.running and self.wake_daemon._ready.is_set())

    def _acknowledge_wake(self) -> None:
        """First spoken message after the wake word, then the listener waits."""

        self.announce("Yes, I am listening. How can I help?")

    def _respond_to_wake_query(self, query: str) -> None:
        if not self.voice_chat_enabled:
            return
        try:
            self.last_reply = self.ollama.ask(query)
            if self.last_reply:
                self.announce(self.last_reply)
        except RuntimeError as exc:
            self.last_reply = str(exc)
            self.announce("I could not reach the language model.")
            logger.warning("Voice chat query failed: %s", exc)

    def close(self) -> None:
        if self.wake_daemon:
            self.wake_daemon.stop()
        self.speaker.close()
