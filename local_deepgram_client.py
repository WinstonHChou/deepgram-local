import asyncio
import json
import uuid
import numpy as np
import sounddevice as sd
import websockets

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

        # Event callbacks
        self.on_open = None
        self.on_interim = None
        self.on_final = None
        self.on_error = None

        self._audio_queue = asyncio.Queue()
        self._seen_completed_ends = set()
        self._last_interim_text = ""
        self._running = False

    def _mic_callback(self, indata, frames, time_info, status):
        """SoundDevice callback: captures float32 mono 16kHz PCM directly."""
        if status:
            print(f"\n[Audio Warning] {status}")
        if self._running:
            mono = indata[:, 0]
            rms = float(np.sqrt(np.mean(mono**2)))

            # Print a lightweight visual meter if no transcript is actively printing
            if not self._last_interim_text:
                bars = int(min(rms * 300, 30))
                meter = "#" * bars + "-" * (30 - bars)
                print(f"\r[Mic Level] |{meter}| RMS: {rms:.4f}", end="", flush=True)

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

            # 1. Handle Server Handshake
            if data.get("message") == "SERVER_READY":
                if self.on_open:
                    self.on_open(data)
                continue

            segments = data.get("segments", [])
            if not segments:
                continue

            # 2. Process segments into Deepgram-style Final vs Interim events
            for seg in segments:
                text = seg.get("text", "").strip()
                if not text:
                    continue

                is_completed = seg.get("completed", False)
                seg_key = (seg.get("start"), seg.get("end"), text)

                if is_completed:
                    # Emit each finalized segment exactly once
                    if seg_key not in self._seen_completed_ends:
                        self._seen_completed_ends.add(seg_key)
                        self._last_interim_text = ""
                        if self.on_final:
                            self.on_final({
                                "is_final": True,
                                "speech_final": True,
                                "transcript": text,
                                "start": float(seg.get("start", 0.0)),
                                "end": float(seg.get("end", 0.0)),
                            })
                else:
                    # Emit interim hypothesis only when text actually changes
                    if text != self._last_interim_text:
                        self._last_interim_text = text
                        if self.on_interim:
                            self.on_interim({
                                "is_final": False,
                                "speech_final": False,
                                "transcript": text,
                                "start": float(seg.get("start", 0.0)),
                                "end": float(seg.get("end", 0.0)),
                            })

    async def start_microphone(self):
        self._loop = asyncio.get_running_loop()
        self._running = True

        async with websockets.connect(self.url) as ws:
            # Send WhisperLive configuration handshake
            config = {
                "uid": self.uid,
                "language": self.language,
                "task": "transcribe",
                "model": self.model,
                "use_vad": self.use_vad,
                "send_last_n_segments": 10,
            }
            await ws.send(json.dumps(config))

            # Wait until OpenVINO finishes compiling/loading the model
            while True:
                resp = json.loads(await ws.recv())
                if resp.get("message") == "SERVER_READY":
                    if self.on_open:
                        self.on_open(resp)
                    break

            # Open microphone stream once server is ready
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


# --- Usage Example ---
if __name__ == "__main__":
    client = LocalDeepgramClient(
        url="ws://localhost:9090",
        model="OpenVINO/whisper-base-fp16-ov",
        language="en",
    )

    client.on_open = lambda meta: print(f"\n[Connected] Backend: {meta.get('backend')} ready. Speak now...")
    client.on_interim = lambda res: print(f"\r[Interim] {res['transcript']}...", end="", flush=True)
    client.on_final = lambda res: print(f"\r[FINAL]   {res['transcript']} ({res['start']:.1f}s - {res['end']:.1f}s)          ")

    try:
        default_in = sd.query_devices(kind='input')
        print(f"[Mic] Using Input Device #{default_in['index']}: {default_in['name']}")
        asyncio.run(client.start_microphone())
    except KeyboardInterrupt:
        print("\nStopped.")