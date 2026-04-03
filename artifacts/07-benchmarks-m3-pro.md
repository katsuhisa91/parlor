# Benchmarks — Apple M3 Pro 18GB

All benchmarks run on LiteRT-LM v0.10.1, compiled from source with Bazel.

## Setup

```bash
# Built binary location
~/workspace/LiteRT-LM/run_dir/litert_lm_main

# IMPORTANT: Must symlink for native Metal backend
cd ~/workspace/LiteRT-LM/run_dir
ln -sf libLiteRtMetalAccelerator.dylib libLiteRtGpuAccelerator.dylib

# Without this symlink, falls back to WebGPU (much slower)
```

## E2B (2.3B effective, 2.58 GB model)

### GPU (Metal) vs CPU

| Metric | GPU Metal | CPU (XNNPACK) | GPU advantage |
|--------|-----------|---------------|---------------|
| **TTFT** | **0.15s** | 0.29s | **1.9x** |
| **Prefill** | **130 tok/s** | 64.4 tok/s | **2.0x** |
| **Decode** | **57.8 tok/s** | 36.7 tok/s | **1.6x** |
| Init time | ~4.1s | ~3s | - |

GPU confirmed via logs:
```
RegisterAccelerator: name=GPU Metal
Created a Metal device.
Created Metal device from provided device id
```

### GPU (Metal) vs GPU (WebGPU fallback)

WebGPU is what you get if `libLiteRtGpuAccelerator.dylib` symlink is missing:

| Metric | Metal Native | WebGPU (Metal backend) |
|--------|-------------|----------------------|
| Decode (E4B) | 26.5 tok/s | 19.7 tok/s |
| TTFT (E4B) | 0.38s | 1.18s |

Always use the Metal symlink.

## E4B (4.5B effective, 3.65 GB model)

### GPU (Metal)

| Metric | Value |
|--------|-------|
| **TTFT** | **0.38s** |
| **Prefill** | **61.3 tok/s** |
| **Decode** | **26.5 tok/s** |
| Init time | ~5.9s |
| 140 token output | 5.3s |

## E2B vs E4B (both Metal GPU)

| Metric | E2B | E4B | E2B advantage |
|--------|-----|-----|---------------|
| **TTFT** | **0.15s** | 0.38s | **2.5x** |
| **Prefill** | **130 tok/s** | 61.3 tok/s | **2.1x** |
| **Decode** | **57.8 tok/s** | 26.5 tok/s | **2.2x** |
| Model size | 2.58 GB | 3.65 GB | 1 GB smaller |

## Comparison with Published Benchmarks

Published benchmarks are E2B on M4 Max:

| Metric | M3 Pro (ours) | M4 Max (published) | Ratio |
|--------|---------------|---------------------|-------|
| Prefill | 130 tok/s | 7,835 tok/s | 0.017x |
| Decode | 57.8 tok/s | 160.2 tok/s | 0.36x |
| TTFT | 0.15s | 0.1s | 1.5x |

Decode ratio (0.36x) roughly matches memory bandwidth ratio (~150 GB/s / ~546 GB/s ≈ 0.27x). Prefill is much lower than expected — likely because M4 Max has significantly more GPU compute cores.

## Estimated Real-Time Pipeline Latency

### E2B (recommended for demo)

| Stage | Time |
|-------|------|
| VAD (silence detection) | ~200ms |
| Multimodal prefill (audio+image ~210 tokens at 130 tok/s) | ~1.6s |
| First sentence decode (~20 tokens at 58 tok/s) | ~0.35s |
| Kokoro TTS first audio | ~0.3s |
| **Total: user stops → AI voice starts** | **~2.5s** |

### E4B

| Stage | Time |
|-------|------|
| VAD (silence detection) | ~200ms |
| Multimodal prefill (audio+image ~210 tokens at 61 tok/s) | ~3.4s |
| First sentence decode (~20 tokens at 27 tok/s) | ~0.75s |
| Kokoro TTS first audio | ~0.3s |
| **Total: user stops → AI voice starts** | **~4.6s** |

## Recommendation

**Use E2B for the demo.** 2.2x faster decode, 2.5x faster TTFT, and the quality difference is acceptable for a real-time conversational demo. The ~2.5s round-trip is comparable to early voice assistants and is a solid proof of concept.

## Multimodal Input — Verified Working

Tested with Python API (CPU), both modalities confirmed functional:

| Modality | Test Input | Model Response | Status |
|----------|-----------|----------------|--------|
| **Image** | 10x10 red JPEG | "The image is primarily **red**." | Working |
| **Audio** | 2s 440Hz sine WAV (16kHz mono) | Processed by audio encoder, generated description | Working |

Python test code:
```python
import litert_lm

with litert_lm.Engine(
    './gemma-4-E2B-it.litertlm',
    backend=litert_lm.Backend.CPU,
    vision_backend=litert_lm.Backend.CPU,
    audio_backend=litert_lm.Backend.CPU
) as engine:
    with engine.create_conversation() as conv:
        # Image
        msg = {"role": "user", "content": [
            {"type": "image", "path": "/tmp/test_image.jpg"},
            {"type": "text", "text": "What color is this image?"}
        ]}
        print(conv.send_message(msg)["content"][0]["text"])

        # Audio
        msg = {"role": "user", "content": [
            {"type": "audio", "path": "/tmp/test_audio.wav"},
            {"type": "text", "text": "Describe the audio."}
        ]}
        print(conv.send_message(msg)["content"][0]["text"])
```

## How to Reproduce

```bash
# E2B GPU benchmark
cd ~/workspace/LiteRT-LM/run_dir
./litert_lm_main --backend=gpu \
  --model_path=./gemma-4-E2B-it.litertlm \
  --input_prompt="Write a detailed paragraph about the ocean."

# E4B GPU benchmark
./litert_lm_main --backend=gpu \
  --model_path=$(find ~/.cache -name "gemma-4-E4B-it.litertlm" -type l | head -1) \
  --input_prompt="Write a detailed paragraph about the ocean."

# CPU benchmark (any model)
./litert_lm_main --backend=cpu \
  --model_path=./gemma-4-E2B-it.litertlm \
  --input_prompt="Write a detailed paragraph about the ocean."
```
