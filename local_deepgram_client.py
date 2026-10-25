import argparse
import asyncio
from collections import deque
import json
import os
import queue
import sys
import threading
import time
import uuid
import numpy as np
import pyperclip
import sounddevice as sd
import websockets
from pynput import keyboard as pynput_kb
from pynput.keyboard import Controller as KeyController, Key


class SystemWideTextInjector:
    """Injects transcribed text into whichever window currently has cursor focus."""

    def __init__(self, live_interim: bool = False):
        self.kb = KeyController()
        self.live_interim = live_interim
        self._active_interim_str = ""
        self._task_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _worker_loop(self):
        while True:
            item = self._task_queue.get()
            if item is None:
                break
            fn, text = item
            try:
                fn(text)
            except Exception:
                pass

    def _clear_interim(self):
        """Erases the currently displayed interim text using Backspaces."""
        if self._active_interim_str:
            for _ in range(len(self._active_interim_str)):
                self.kb.press(Key.backspace)
                self.kb.release(Key.backspace)
            self._active_interim_str = ""

    def _paste_fast(self, text: str):
        """Pastes text via clipboard (fastest & most reliable across IDEs/Apps) and restores clipboard."""
        try:
            old_clip = pyperclip.paste()
        except Exception:
            old_clip = None

        pyperclip.copy(text)
        time.sleep(0.01)

        with self.kb.pressed(Key.ctrl):
            self.kb.press("v")
            self.kb.release("v")

        if old_clip is not None and old_clip != text:
            time.sleep(0.02)
            try:
                pyperclip.copy(old_clip)
            except Exception:
                pass

    def _do_interim(self, text: str):
        common_len = len(os.path.commonprefix((self._active_interim_str, text)))
        backspaces_needed = len(self._active_interim_str) - common_len
        for _ in range(backspaces_needed):
            self.kb.press(Key.backspace)
            self.kb.release(Key.backspace)

        new_suffix = text[common_len:]
        if new_suffix:
            self.kb.type(new_suffix)

        self._active_interim_str = text

    def _do_final(self, text: str):
        if self.live_interim and self._active_interim_str:
            self._clear_interim()

        # Trailing space so consecutive sentences don't mash together
        out = text.strip() + " "
        if self.live_interim:
            self.kb.type(out)
        else:
            self._paste_fast(out)

    def handle_interim(self, text: str):
        """Optional: live ghost-typing with minimal backspace diffing."""
        if not self.live_interim:
            return
        self._task_queue.put((self._do_interim, text))

    def handle_final(self, text: str):
        """Commits finalized text to the active cursor position without blocking the event loop."""
        self._task_queue.put((self._do_final, text))


def _iter_input_devices() -> list:
    """Return every host device that can capture audio, each with an explicit 'index'."""
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        info = dict(dev)
        info["index"] = int(info.get("index", idx))
        if int(info.get("max_input_channels", 0) or 0) > 0:
            devices.append(info)
    return devices


def resolve_input_device(device=None) -> dict:
    """Resolve None | device index | name fragment to an input device dict.

    Fails loudly (LookupError) on unknown/ambiguous selectors so the caller can
    report it before doing any expensive setup.
    """
    if device is None:
        # Raises ValueError when the machine has no input device at all.
        return dict(sd.query_devices(kind="input"))

    devices = _iter_input_devices()
    if not devices:
        raise LookupError(
            "No audio input devices found. Check the microphone connection and OS permissions."
        )

    selector = str(device).strip()
    if selector.lstrip("+").isdigit():
        wanted = int(selector)
        for dev in devices:
            if dev["index"] == wanted:
                return dev
        raise LookupError(
            f"No input device with index {wanted}. "
            f"Available: {_format_device_list(devices)}. Use --list-devices for details."
        )

    needle = selector.casefold()
    matches = [dev for dev in devices if needle in str(dev["name"]).casefold()]
    if len(matches) > 1:
        raise LookupError(
            f"{device!r} matches multiple input devices ({_format_device_list(matches)}). "
            "Disambiguate with a device index."
        )
    if not matches:
        raise LookupError(
            f"No input device matching {device!r}. "
            f"Available: {_format_device_list(devices)}. Use --list-devices for details."
        )
    return matches[0]


def validate_input_channel(device_info: dict, channel: int) -> int:
    """Clamp-check a requested input channel against the device's channel count."""
    max_channels = int(device_info.get("max_input_channels", 1) or 1)
    if not 0 <= channel < max_channels:
        raise LookupError(
            f"Input device #{device_info.get('index')} ({device_info.get('name')}) exposes "
            f"{max_channels} input channel(s); --channel {channel} is out of range "
            f"(valid: 0-{max_channels - 1})."
        )
    return channel


def _format_device_list(devices) -> str:
    return ", ".join(f"#{d['index']}: {d['name']}" for d in devices)


def list_input_devices() -> None:
    """Print all capture devices, marking the system default with a '*'."""
    devices = _iter_input_devices()
    if not devices:
        print("No audio input devices found (check connections and OS mic permissions).")
        return

    try:
        default_index = int(sd.query_devices(kind="input")["index"])
    except Exception:
        default_index = None

    print("Available input devices (* = system default):")
    for dev in devices:
        marker = "*" if dev["index"] == default_index else " "
        try:
            hostapi = sd.query_hostapi(dev["hostapi"])["name"]
        except Exception:
            hostapi = ""
        print(
            f"  {marker} #{dev['index']:<3} {dev['name']}"
            f"  [{hostapi}] channels={dev['max_input_channels']}"
            f" rate={dev.get('default_samplerate')}"
        )
    print("\nSelect one with --device INDEX or --device 'name fragment' (--channel N for multi-channel gear).")


class LocalDeepgramClient:
    def __init__(
        self,
        url: str = "ws://localhost:9090",
        model: str = "OpenVINO/whisper-base-fp16-ov",
        language: str = "en",
        use_vad: bool = True,
        sample_rate: int = 16000,
        chunk_size: int = 2048,
        verbose: bool = False,
        max_queue_chunks: int = 64,
        max_seen_segments: int = 64,
        device=None,
        channel: int = 0,
    ):
        self.url = url
        self.model = model
        self.language = language
        self.use_vad = use_vad
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.device = device
        self.channel = channel
        self._channel_extractor = None  # set by start_microphone() when channel > 0
        self.uid = str(uuid.uuid4())

        self.on_open = None
        self.on_interim = None
        self.on_final = None

        self._loop = None
        self._audio_queue = asyncio.Queue(maxsize=max_queue_chunks)
        self._seen_completed_ends = set()
        self._seen_completed_order = deque(maxlen=max_seen_segments)
        self._last_interim_text = ""
        self._running = False
        self.dictation_enabled = True  # Toggled via global hotkey

    def _log(self, message: str, **kwargs):
        if self.verbose:
            print(message, **kwargs)

    def _clear_audio_queue(self):
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _enqueue_audio(self, pcm_bytes: bytes):
        if not (self._running and self.dictation_enabled):
            return
        if self._audio_queue.full():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self._audio_queue.put_nowait(pcm_bytes)
        except asyncio.QueueFull:
            pass

    def toggle_dictation(self):
        self.dictation_enabled = not self.dictation_enabled
        if not self.dictation_enabled:
            self._last_interim_text = ""
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._clear_audio_queue)
        state = "ON (Listening & Typing)" if self.dictation_enabled else "MUTED (Paused)"
        print(f"\n[Hotkey F8] Dictation is now: {state}")

    def _build_channel_extractor(self, wanted_channel: int, capture_channels: int):
        """Return a callable that pulls one channel out of interleaved float32 PCM.

        Returns None for the plain mono case so the PortAudio callback stays
        NumPy-free. Input-only streams cannot select a channel via a keyword
        argument, so we capture 0..wanted_channel and de-interleave ourselves.
        """
        if wanted_channel == 0 and capture_channels == 1:
            return None

        def extract(indata):
            frames = np.frombuffer(indata, dtype=np.float32)
            remainder = frames.size % capture_channels
            if remainder:
                frames = frames[: frames.size - remainder]
            return frames.reshape(-1, capture_channels)[:, wanted_channel].tobytes()

        return extract

    def _mic_callback(self, indata, frames, time_info, status):
        if not (self._running and self.dictation_enabled and self._loop is not None):
            return
        # indata is only valid for the duration of this callback -> copy it here
        payload = bytes(indata) if self._channel_extractor is None else self._channel_extractor(indata)
        if payload:
            self._loop.call_soon_threadsafe(self._enqueue_audio, payload)

    async def _send_audio_loop(self, ws):
        while self._running:
            chunk = await self._audio_queue.get()
            if chunk is None:
                break

            # Coalesce any immediately queued chunks into a single WebSocket send
            if not self._audio_queue.empty():
                chunks = [chunk]
                while not self._audio_queue.empty():
                    try:
                        next_chunk = self._audio_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if next_chunk is None:
                        await ws.send(b"".join(chunks))
                        return
                    chunks.append(next_chunk)
                chunk = b"".join(chunks)

            await ws.send(chunk)

    def _record_completed_segment(self, seg_key) -> bool:
        """Returns True if seg_key is newly completed, maintaining a bounded set."""
        if seg_key in self._seen_completed_ends:
            return False
        if len(self._seen_completed_order) == self._seen_completed_order.maxlen:
            oldest = self._seen_completed_order.popleft()
            self._seen_completed_ends.discard(oldest)
        self._seen_completed_order.append(seg_key)
        self._seen_completed_ends.add(seg_key)
        return True

    async def _recv_transcript_loop(self, ws):
        async for raw_msg in ws:
            data = json.loads(raw_msg)

            segments = data.get("segments")
            if segments is None:
                if data.get("message") == "SERVER_READY" and self.on_open:
                    self.on_open(data)
                continue

            if not segments or not self.dictation_enabled:
                continue

            for seg in segments:
                is_completed = seg.get("completed", False)
                if not is_completed and self.on_interim is None:
                    continue

                text = seg.get("text", "").strip()
                if not text:
                    continue

                if is_completed:
                    seg_key = (seg.get("start"), seg.get("end"), text)
                    if self._record_completed_segment(seg_key):
                        self._last_interim_text = ""
                        if self.on_final:
                            self.on_final({
                                "is_final": True,
                                "transcript": text,
                                "start": float(seg.get("start", 0.0)),
                                "end": float(seg.get("end", 0.0)),
                            })
                else:
                    if text != self._last_interim_text:
                        self._last_interim_text = text
                        self.on_interim({
                            "is_final": False,
                            "transcript": text,
                            "start": float(seg.get("start", 0.0)),
                            "end": float(seg.get("end", 0.0)),
                        })

    async def start_microphone(self):
        self._loop = asyncio.get_running_loop()
        self._running = True

        # Resolve the microphone *before* touching the network: a missing,
        # unplugged, or ambiguous device now fails immediately instead of after
        # the WebSocket handshake and model-server warm-up.
        device_info = resolve_input_device(self.device)
        channel = validate_input_channel(device_info, self.channel)
        capture_channels = channel + 1
        self._channel_extractor = self._build_channel_extractor(channel, capture_channels)

        max_channels = int(device_info.get("max_input_channels", 1) or 1)
        extra = f", capturing {capture_channels} of them to reach it" if capture_channels > 1 else ""
        print(
            f"[Mic] Using Input Device #{device_info['index']}: {device_info['name']}"
            f" (input channel {channel} of {max_channels}{extra})"
        )

        async with websockets.connect(self.url, compression=None) as ws:
            config = {
                "uid": self.uid,
                "language": self.language,
                "task": "transcribe",
                "model": self.model,
                "use_vad": self.use_vad,
                "send_last_n_segments": 10,
                "same_output_threshold": 5,  # Faster finalization for live dictation
            }
            await ws.send(json.dumps(config))

            while True:
                resp = json.loads(await ws.recv())
                if resp.get("message") == "SERVER_READY":
                    if self.on_open:
                        self.on_open(resp)
                    break

            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=capture_channels,
                dtype="float32",
                device=device_info["index"],
                callback=self._mic_callback,
            ):
                await asyncio.gather(
                    self._send_audio_loop(ws),
                    self._recv_transcript_loop(ws),
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Deepgram-style dictation client")
    parser.add_argument("--url", default="ws://localhost:9090", help="WebSocket server URL")
    parser.add_argument("--model", default="OpenVINO/whisper-base-fp16-ov", help="Whisper model name")
    parser.add_argument("--language", default="en", help="Language code")
    parser.add_argument("--live-interim", action="store_true", help="Type and backspace interim results live")
    parser.add_argument(
        "--device",
        default=None,
        help="Input device index or (case-insensitive) name fragment. Defaults to the system default input.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Input channel to capture from the selected device (default: 0)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available input devices and exit",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable console status and transcript printouts")
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        sys.exit(0)

    injector = SystemWideTextInjector(live_interim=args.live_interim)

    client = LocalDeepgramClient(
        url=args.url,
        model=args.model,
        language=args.language,
        verbose=args.verbose,
        device=args.device,
        channel=args.channel,
    )

    # Register global F8 hotkey to toggle dictation on/off from any window
    hotkey_listener = pynput_kb.GlobalHotKeys({
        "<f8>": client.toggle_dictation,
    })
    hotkey_listener.start()

    def on_open(meta):
        print(f"\n[Connected] Backend: {meta.get('backend')} ready.")
        print("[Controls] Press F8 anywhere to pause/resume dictation. Click any text box and speak!")

    def on_interim(res):
        if args.verbose:
            print(f"\r[Interim] {res['transcript']}...", end="", flush=True)
        injector.handle_interim(res["transcript"])

    def on_final(res):
        if args.verbose:
            print(f"\r[TYPED]   {res['transcript']}                              ")
        injector.handle_final(res["transcript"])

    client.on_open = on_open
    client.on_interim = on_interim if (args.verbose or args.live_interim) else None
    client.on_final = on_final

    try:
        asyncio.run(client.start_microphone())
    except (LookupError, ValueError, OSError) as exc:
        hotkey_listener.stop()
        print(f"\n[Error] {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        hotkey_listener.stop()
        print("\nStopped.")