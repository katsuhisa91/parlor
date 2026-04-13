"""Parlor — on-device, real-time multimodal AI (voice + vision) — mlx backend."""

import asyncio
import base64
import json
import os
import re
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import tts

VLM_MODEL = os.environ.get("VLM_MODEL", "mlx-community/gemma-4-E2B-it-4bit")

TOOLS = {
    "get_current_time": {
        "description": "現在の日時を取得する",
        "fn": lambda: __import__("datetime").datetime.now().strftime("%Y年%m月%d日 %H時%M分"),
    },
}

SYSTEM_PROMPT = (
    "あなたは親しみやすいAIアシスタントです。"
    "ユーザーはマイクを通して話しかけています。"
    "必ず日本語で返答してください。"
    "返答は短く自然に：1〜3文で。\n\n"
    "## ツール\n"
    "リアルタイム情報が必要な場合は以下のJSON形式のみで返答し、他のテキストを含めないでください:\n"
    '{"tool_call": {"name": "<ツール名>", "arguments": {}}}\n'
    "利用可能なツール:\n"
    "- get_current_time: 現在の日時を返す（引数なし）\n"
)

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

_vlm_model = None
_vlm_processor = None
_vlm_config = None
_tts_backend = None


def load_models():
    global _vlm_model, _vlm_processor, _vlm_config, _tts_backend

    print(f"Loading Gemma 4 E2B ({VLM_MODEL})...")
    from mlx_vlm import load as vlm_load
    from mlx_vlm.utils import load_config as vlm_load_config
    _vlm_model, _vlm_processor = vlm_load(VLM_MODEL)
    _vlm_config = vlm_load_config(VLM_MODEL)
    print("Gemma 4 E2B loaded.")

    _tts_backend = tts.load()


@asynccontextmanager
async def lifespan(app):
    await asyncio.get_event_loop().run_in_executor(None, load_models)
    yield


app = FastAPI(lifespan=lifespan)


def _transcribe(wav_b64: str) -> str:
    """Gemma 4 E2B の内蔵 ASR で音声認識を行う。"""
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    wav_bytes = base64.b64decode(wav_b64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp_path = f.name
    try:
        prompt = apply_chat_template(
            _vlm_processor, _vlm_config,
            "この音声を正確に文字起こしして、日本語で返してください。",
            num_audios=1,
        )
        result = vlm_generate(
            _vlm_model, _vlm_processor,
            prompt,
            audio=[tmp_path],
            max_tokens=300,
            verbose=False,
        )
        return (result.text if hasattr(result, "text") else str(result)).strip()
    finally:
        os.unlink(tmp_path)


def _parse_tool_call(text: str):
    """ツール呼び出しのJSONをテキスト内から検出して返す。なければNone。"""
    for m in re.finditer(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text):
        try:
            obj = json.loads(m.group(1))
            if "tool_call" in obj:
                return obj["tool_call"]
        except json.JSONDecodeError:
            pass

    for start in (m.start() for m in re.finditer(r'\{', text)):
        depth = 0
        for i, ch in enumerate(text[start:]):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:start + i + 1])
                        if "tool_call" in obj:
                            return obj["tool_call"]
                    except json.JSONDecodeError:
                        pass
                    break
    return None


def _execute_tool(name: str, arguments: dict) -> str:
    if name not in TOOLS:
        return f"エラー: ツール '{name}' は存在しません"
    try:
        return str(TOOLS[name]["fn"](**arguments))
    except Exception as e:
        return f"エラー: {e}"


def _llm_generate_once(conversation: list, image_b64: str = None) -> str:
    """Gemma 4 E2B で LLM 推論を行う（画像対応）。"""
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    images = None
    num_images = 0

    if image_b64:
        import base64 as _b64
        import io as _io
        from PIL import Image as _PILImage
        images = [_PILImage.open(_io.BytesIO(_b64.b64decode(image_b64)))]
        num_images = 1

    prompt = apply_chat_template(
        _vlm_processor, _vlm_config,
        conversation,
        num_images=num_images,
    )

    result = vlm_generate(
        _vlm_model, _vlm_processor,
        prompt,
        image=images,
        max_tokens=128,
        verbose=False,
    )
    return (result.text if hasattr(result, "text") else str(result)).strip()


def _llm_respond(conversation: list, image_b64: str = None, max_tool_rounds: int = 3) -> str:
    """ツール呼び出しループ付きのLLM応答。"""
    conv = list(conversation)

    for round_idx in range(max_tool_rounds + 1):
        response = _llm_generate_once(conv, image_b64 if round_idx == 0 else None)

        tool_call = _parse_tool_call(response)
        if tool_call is None:
            return response

        name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})
        tool_result = _execute_tool(name, arguments)
        print(f"Tool call: {name}({arguments}) → {tool_result}")

        conv.append({"role": "assistant", "content": response})
        conv.append({"role": "user", "content": f"[ツール結果: {name}] {tool_result}"})

    return response


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
                transcription = await asyncio.get_event_loop().run_in_executor(
                    None, _transcribe, msg["audio"]
                )
                print(f"ASR ({time.time()-t0:.2f}s): {transcription!r}")

            user_text = transcription or msg.get("text", "Hello!")
            if not user_text:
                continue

            image_b64 = msg.get("image")

            conversation.append({"role": "user", "content": user_text})

            t1 = time.time()
            conv_snapshot = list(conversation)
            text_response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: _llm_respond(conv_snapshot, image_b64)
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
