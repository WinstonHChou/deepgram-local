import asyncio
import json
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
            old_clip = ""

        pyperclip.copy(text)
        time.sleep(0.01)

        with self.kb.pressed(Key.ctrl):
            self.kb.press("v")
            self.kb.release("v")

        time.sleep(0.02)
        try:
            pyperclip.copy(old_clip)
        except Exception:
            pass

    def handle_interim(self, text: str):
        """Optional: live ghost-typing with minimal backspace diffing."""
        if not self.live_interim:
            return

        # Find common prefix so we only backspace the characters that changed
        common_len = 0
        for c1, c2 in zip(self._active_interim_str, text):
            if c1 == c2:
                common_len += 1
            else:
                break

        backspaces_needed = len(self._active_interim_str) - common_len
        for _ in range(backspaces_needed):
            self.kb.press(Key.backspace)
            self.kb.release(Key.backspace)

        new_suffix = text[common_len:]
        if new_suffix:
            self.kb.type(new_suffix)

        self._active_interim_str = text

    def handle_final(self, text: str):
        """Commits finalized text to the active cursor position."""
        if self.live_interim and self._active_interim_str:
            self._clear_interim()

        # Trailing space so consecutive sentences don't mash together
        out = text.strip() + " "
        if self.live_interim:
            self.kb.type(out)
        else:
            self._paste_fast(out)


class LocalDeepgramClient:
    def __init__(
        self,
        url: str = "ws://localhost:9090",
        model: str = "OpenVINO/whisper-base-fp16-ov",
        language: str = "en",
        use_vad: bool = True,
        sample_rate: int = 16000,
        chunk_size: int = 2048,
    ):
        self.url = url
        self.model = model
        self.language = language
        self.use_vad = use_vad
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.uid = str(uuid.uuid4())

        self.on_open = None
        self.on_interim = None
        self.on_final = None

        self._audio_queue = asyncio.Queue()
        self._seen_completed_ends = set()
        self._last_interim_text = ""
        self._running = False
        self.dictation_enabled = True  # Toggled via global hotkey

    def toggle_dictation(self):
        self.dictation_enabled = not self.dictation_enabled
        state = "ON (Listening & Typing)" if self.dictation_enabled else "MUTED (Paused)"
        print(f"\n[Hotkey F8] Dictation is now: {state}")

    def _mic_callback(self, indata, frames, time_info, status):
        if self._running and self.dictation_enabled:
            mono = indata[:, 0]
            pcm_float32_bytes = mono.astype(np.float32).tobytes()
            self._loop.call_soon_threadsafe(self._audio_queue.put_nowait, pcm_float32_bytes)

    async def _send_audio_loop(self, ws):
        while self._running:
            chunk = await self._audio_queue.get()
            if chunk is None:
                break
            await ws.send(chunk)

    async def _recv_transcript_loop(self, ws):
        async for raw_msg in ws:
            data = json.loads(raw_msg)

            if data.get("message") == "SERVER_READY":
                if self.on_open:
                    self.on_open(data)
                continue

            segments = data.get("segments", [])
            if not segments or not self.dictation_enabled:
                continue

            for seg in segments:
                text = seg.get("text", "").strip()
                if not text:
                    continue

                is_completed = seg.get("completed", False)
                seg_key = (seg.get("start"), seg.get("end"), text)

                if is_completed:
                    if seg_key not in self._seen_completed_ends:
                        self._seen_completed_ends.add(seg_key)
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
                        if self.on_interim:
                            self.on_interim({
                                "is_final": False,
                                "transcript": text,
                                "start": float(seg.get("start", 0.0)),
                                "end": float(seg.get("end", 0.0)),
                            })

    async def start_microphone(self):
        self._loop = asyncio.get_running_loop()
        self._running = True

        async with websockets.connect(self.url) as ws:
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

            default_in = sd.query_devices(kind="input")
            print(f"[Mic] Using Input Device #{default_in['index']}: {default_in['name']}")

            with sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=1,
                dtype="float32",
                callback=self._mic_callback,
            ):
                await asyncio.gather(
                    self._send_audio_loop(ws),
                    self._recv_transcript_loop(ws),
                )


if __name__ == "__main__":
    # Set live_interim=True if you want words typed and backspaced in real-time before finalization
    injector = SystemWideTextInjector(live_interim=False)

    client = LocalDeepgramClient(
        url="ws://localhost:9090",
        model="OpenVINO/whisper-base-fp16-ov",
        language="en",
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
        print(f"\r[Interim] {res['transcript']}...", end="", flush=True)
        injector.handle_interim(res["transcript"])

    def on_final(res):
        print(f"\r[TYPED]   {res['transcript']}                              ")
        injector.handle_final(res["transcript"])

    client.on_open = on_open
    client.on_interim = on_interim
    client.on_final = on_final

    try:
        asyncio.run(client.start_microphone())
    except KeyboardInterrupt:
        hotkey_listener.stop()
        print("\nStopped.")