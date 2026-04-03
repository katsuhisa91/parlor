"""Real-time multimodal AI demo with Gemma 4 E2B + Kokoro TTS."""

import asyncio
import base64
import io
import json
import os
import struct
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import litert_lm

# -- Config --
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.expanduser("~/workspace/LiteRT-LM/run_dir/gemma-4-E2B-it.litertlm"),
)

SAMPLE_RATE = 24000  # Kokoro output sample rate
INPUT_SAMPLE_RATE = 16000  # LiteRT-LM audio input sample rate

SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. The user is talking to you "
    "through a microphone and showing you their camera. Respond naturally and "
    "concisely in 1-3 short sentences. Be direct and conversational."
)

app = FastAPI()

# -- Global engine (loaded once) --
engine = None
kokoro_model = None


def get_engine():
    global engine
    if engine is None:
        print(f"Loading Gemma 4 E2B from {MODEL_PATH}...")
        engine = litert_lm.Engine(
            MODEL_PATH,
            backend=litert_lm.Backend.CPU,
            vision_backend=litert_lm.Backend.CPU,
            audio_backend=litert_lm.Backend.CPU,
        )
        engine.__enter__()
        print("Engine loaded.")
    return engine


def get_kokoro():
    global kokoro_model
    if kokoro_model is None:
        print("Loading Kokoro TTS...")
        import kokoro_onnx
        tts_dir = Path(__file__).parent
        kokoro_model = kokoro_onnx.Kokoro(
            str(tts_dir / "kokoro-v1.0.onnx"),
            str(tts_dir / "voices-v1.0.bin"),
        )
        print("Kokoro TTS loaded.")
    return kokoro_model


def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 PCM numpy array to WAV bytes."""
    # Normalize to int16
    pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


def save_temp_wav(audio_bytes: bytes) -> str:
    """Save raw audio bytes (already WAV format from browser) to temp file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name


def save_temp_image(image_bytes: bytes) -> str:
    """Save JPEG bytes to temp file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.write(image_bytes)
    tmp.close()
    return tmp.name


@app.on_event("startup")
async def startup():
    # Pre-load models in background
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_engine)
    # Kokoro downloads models on first use, do it eagerly
    await loop.run_in_executor(None, get_kokoro)


@app.get("/")
async def root():
    return HTMLResponse(content=open(Path(__file__).parent / "index.html").read())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    eng = get_engine()
    tts = get_kokoro()

    # Create a fresh conversation with system prompt
    conversation = eng.create_conversation(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}]
    )
    conversation.__enter__()

    try:
        while True:
            # Receive message from browser
            raw = await ws.receive_text()
            msg = json.loads(raw)

            audio_path = None
            image_path = None

            try:
                # Save audio to temp WAV file
                if msg.get("audio"):
                    audio_bytes = base64.b64decode(msg["audio"])
                    audio_path = save_temp_wav(audio_bytes)

                # Save image to temp JPEG file
                if msg.get("image"):
                    image_bytes = base64.b64decode(msg["image"])
                    image_path = save_temp_image(image_bytes)

                # Build multimodal message
                content = []
                if audio_path:
                    content.append({"type": "audio", "path": os.path.abspath(audio_path)})
                if image_path:
                    content.append({"type": "image", "path": os.path.abspath(image_path)})

                # Add text instruction
                if audio_path and image_path:
                    content.append({
                        "type": "text",
                        "text": "The user just spoke to you (audio) while showing their camera (image). Respond to what they said, referencing what you see if relevant.",
                    })
                elif audio_path:
                    content.append({
                        "type": "text",
                        "text": "The user just spoke to you. Respond to what they said.",
                    })
                elif image_path:
                    content.append({
                        "type": "text",
                        "text": "The user is showing you their camera. Describe what you see.",
                    })
                else:
                    # Text-only fallback
                    content.append({"type": "text", "text": msg.get("text", "Hello!")})

                user_msg = {"role": "user", "content": content}

                # Send to LLM (blocking, run in executor)
                t0 = time.time()

                def run_inference():
                    return conversation.send_message(user_msg)

                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, run_inference)
                t1 = time.time()

                text_response = response["content"][0]["text"]
                print(f"LLM ({t1-t0:.2f}s): {text_response}")

                # Send text to client immediately
                await ws.send_text(json.dumps({
                    "type": "text",
                    "text": text_response,
                    "llm_time": round(t1 - t0, 2),
                }))

                # Generate TTS audio
                t2 = time.time()

                def run_tts():
                    samples, sr = tts.create(
                        text_response,
                        voice="af_heart",
                        speed=1.1,
                    )
                    return pcm_to_wav_bytes(samples, sr)

                wav_bytes = await loop.run_in_executor(None, run_tts)
                t3 = time.time()
                print(f"TTS ({t3-t2:.2f}s): {len(wav_bytes)} bytes")

                # Send audio to client
                await ws.send_text(json.dumps({
                    "type": "audio",
                    "audio": base64.b64encode(wav_bytes).decode(),
                    "tts_time": round(t3 - t2, 2),
                }))

            finally:
                # Cleanup temp files
                if audio_path and os.path.exists(audio_path):
                    os.unlink(audio_path)
                if image_path and os.path.exists(image_path):
                    os.unlink(image_path)

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        conversation.__exit__(None, None, None)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
