"""Parlor — on-device, real-time multimodal AI (voice + vision) — mlx backend."""

import asyncio
import base64
import io
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import tts

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "mlx-community/whisper-medium-mlx-4bit")
LLM_MODEL = os.environ.get("LLM_MODEL", "mlx-community/gemma-3-4b-it-4bit")

SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. The user is talking to you through a microphone. "
    "Keep your responses short and natural: 1-3 sentences."
)

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

_lm_model = None
_lm_tokenizer = None
_tts_backend = None


def load_models():
    global _lm_model, _lm_tokenizer, _tts_backend

    print(f"Loading Whisper ({WHISPER_MODEL})...")
    import mlx_whisper
    mlx_whisper.transcribe(
        np.zeros(3200, dtype=np.float32),
        path_or_hf_repo=WHISPER_MODEL,
        verbose=False,
    )
    print("Whisper loaded.")

    print(f"Loading LLM ({LLM_MODEL})...")
    from mlx_lm import load as lm_load
    _lm_model, _lm_tokenizer = lm_load(LLM_MODEL)
    print("LLM loaded.")

    _tts_backend = tts.load()


@asynccontextmanager
async def lifespan(app):
    await asyncio.get_event_loop().run_in_executor(None, load_models)
    yield


app = FastAPI(lifespan=lifespan)


def _decode_wav(wav_b64: str) -> np.ndarray:
    wav_bytes = base64.b64decode(wav_b64)
    audio, _ = sf.read(io.BytesIO(wav_bytes))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def _transcribe(audio: np.ndarray) -> str:
    import mlx_whisper
    result = mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER_MODEL, verbose=False)
    return result["text"].strip()


def _llm_generate(conversation: list) -> str:
    from mlx_lm import generate as lm_generate
    prompt = _lm_tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    return lm_generate(_lm_model, _lm_tokenizer, prompt=prompt, max_tokens=128, verbose=False).strip()


def split_sentences(text: str) -> list:
    parts = SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    interrupted = asyncio.Event()
    msg_queue = asyncio.Queue()

    async def receiver():
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "interrupt":
                    interrupted.set()
                    print("Client interrupted")
                else:
                    await msg_queue.put(msg)
        except WebSocketDisconnect:
            await msg_queue.put(None)

    recv_task = asyncio.create_task(receiver())

    try:
        while True:
            msg = await msg_queue.get()
            if msg is None:
                break

            interrupted.clear()

            transcription = ""
            if msg.get("audio"):
                t0 = time.time()
                audio = _decode_wav(msg["audio"])
                transcription = await asyncio.get_event_loop().run_in_executor(
                    None, _transcribe, audio
                )
                print(f"ASR ({time.time()-t0:.2f}s): {transcription!r}")

            user_text = transcription or msg.get("text", "Hello!")
            if not user_text:
                continue

            conversation.append({"role": "user", "content": user_text})

            t1 = time.time()
            conv_snapshot = list(conversation)
            text_response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: _llm_generate(conv_snapshot)
            )
            llm_time = time.time() - t1
            print(f"LLM ({llm_time:.2f}s): {transcription!r} → {text_response}")

            conversation.append({"role": "assistant", "content": text_response})

            if interrupted.is_set():
                print("Interrupted after LLM, skipping response")
                continue

            reply = {
                "type": "text",
                "text": text_response,
                "llm_time": round(llm_time, 2),
            }
            if transcription:
                reply["transcription"] = transcription
            await ws.send_text(json.dumps(reply))

            if interrupted.is_set():
                print("Interrupted before TTS, skipping audio")
                continue

            sentences = split_sentences(text_response)
            if not sentences:
                sentences = [text_response]

            tts_start = time.time()
            await ws.send_text(json.dumps({
                "type": "audio_start",
                "sample_rate": _tts_backend.sample_rate,
                "sentence_count": len(sentences),
            }))

            for i, sentence in enumerate(sentences):
                if interrupted.is_set():
                    print(f"Interrupted during TTS (sentence {i+1}/{len(sentences)})")
                    break

                pcm = await asyncio.get_event_loop().run_in_executor(
                    None, lambda s=sentence: _tts_backend.generate(s)
                )

                if interrupted.is_set():
                    break

                pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
                await ws.send_text(json.dumps({
                    "type": "audio_chunk",
                    "audio": base64.b64encode(pcm_int16.tobytes()).decode(),
                    "index": i,
                }))

            tts_time = time.time() - tts_start
            print(f"TTS ({tts_time:.2f}s): {len(sentences)} sentences")

            if not interrupted.is_set():
                await ws.send_text(json.dumps({
                    "type": "audio_end",
                    "tts_time": round(tts_time, 2),
                }))

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        recv_task.cancel()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
