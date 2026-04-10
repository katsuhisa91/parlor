# Parlor (日本語対応・MLX版)

オンデバイスでリアルタイムに動く、マルチモーダルAIとの音声・カメラ会話システム。
**Apple Silicon Mac** 上で全モデルをローカル実行します。

このリポジトリは [fikrikarim/parlor](https://github.com/fikrikarim/parlor) をフォークし、
**日本語対応** と **MLXバックエンドへの切替** を中心に改変したものです。

---

## 開発の経緯（裏話）

もともと **Gemma 4** の学習用として始めたリポジトリでした。

Gemma 4 は Google が2025年にリリースした最新マルチモーダルモデルで、
ブラウザ上の WebGPU でも動かせる小型モデル（E2B）が存在します。
元のParlor はこれを LiteRT-LM で動かす構成になっていました。

ところが実際に触ってみると、Gemma 4 E2B を動かすには
**LiteRT-LM のインストールが複雑**で、
日本語対応・ツール呼び出し・カメラ対応をすべて組み合わせようとすると
なかなかうまくいきませんでした。

そこで Apple Silicon に最適化された **MLXバックエンド（mlx-lm）** に切り替えたところ、
動作が安定しました。その結果、使うモデルが **Gemma 3 4B** になってしまい、
「Gemma 4 を学ぶはずがいつの間にか Gemma 3 を動かしていた」という
やや本末転倒な展開になっています。

---

## 元のParlor からの主な差分

| 項目 | 元のParlor | このフォーク |
|------|-----------|-------------|
| 推論バックエンド | LiteRT-LM (Google) | MLX (Apple Silicon) |
| 言語モデル | Gemma 4 E2B | Gemma 3 4B 4bit |
| 音声認識 (ASR) | LiteRT内蔵 (Gemma 4) | mlx-whisper (whisper-medium-4bit) |
| TTS言語 | 英語 (af_heart) | 日本語 (jf_alpha) |
| フォネマイザ | misaki[en] | misaki[en,ja] + pyopenjtalk |
| ツール呼び出し | Gemma 4 ネイティブ function calling | JSON パース + サーバー側実行ループ |
| ビジョン | Gemma 4 マルチモーダル | Phi-3.5-vision-instruct-4bit (VLM) |

### アーキテクチャ

```
Browser (マイク + カメラ)
    │
    │  WebSocket (WAV音声 + JPEG フレーム)
    ▼
FastAPI サーバー
    ├── mlx-whisper (whisper-medium-4bit)   →  日本語音声認識
    ├── Gemma 3 4B 4bit via mlx-lm          →  テキスト生成 + ツール呼び出し
    ├── Phi-3.5-vision-instruct-4bit (VLM)  →  カメラ映像の説明生成
    └── Kokoro TTS (mlx-audio / jf_alpha)   →  日本語音声合成
    │
    │  WebSocket (ストリーミング音声チャンク)
    ▼
Browser (再生 + 文字起こし表示)
```

### ツール呼び出しの仕組み

mlx-lm は Gemma 4 のようなネイティブ function calling をサポートしていないため、
LLM に JSON フォーマットで出力させ、サーバー側でパースして実行するループを実装しています。

```
ユーザー: 「今何時？」
  ↓
Gemma 3: {"tool_call": {"name": "get_current_time", "arguments": {}}}
  ↓
サーバー: datetime.now() を実行 → "2026年04月11日 01時47分"
  ↓
Gemma 3: "今は2026年4月11日の1時47分ですよ！"
  ↓
Kokoro TTS: 音声を生成して返す
```

### カメラ映像の処理

Gemma 3 (mlx-lm) はテキスト専用のため、カメラ映像は別のVLMで処理します。

```
カメラ映像 (JPEG)
  ↓
Phi-3.5-vision-instruct-4bit
  ↓ 「本棚の前にマイクが立っている」（1文の説明）
  ↓
[カメラ映像: 本棚の前にマイクが立っている] ← Gemma 3 のプロンプトに挿入
  ↓
Gemma 3: 「本棚の前にいるんですね！」
```

---

## セットアップ

### 必要環境

- Python 3.12
- macOS + Apple Silicon (M1/M2/M3)
- 空きメモリ: ~6GB（Gemma 3 2.6GB + Phi-3.5-vision 2.3GB + Whisper）

### インストール

```bash
git clone https://github.com/katsuhisa91/parlor.git
cd parlor/src

# uv をインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係をインストール
LDFLAGS="-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib" \
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk \
uv sync

# 日本語辞書のダウンロード（日本語TTS に必要）
uv run python -m unidic download
```

> **Note:** `pyopenjtalk` のビルドに `LDFLAGS` / `SDKROOT` の指定が必要です。
> SDK のパスは `xcrun --show-sdk-path` で確認してください。

### 起動

```bash
LDFLAGS="-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib" \
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk \
uv run server.py
```

[http://localhost:8000](http://localhost:8000) を開いてカメラとマイクを許可し、話しかけてください。

初回起動時にモデルが自動ダウンロードされます（合計 ~6GB）。

---

## 環境変数

| 変数名 | デフォルト | 説明 |
|--------|-----------|------|
| `WHISPER_MODEL` | `mlx-community/whisper-medium-mlx-4bit` | Whisper モデル |
| `LLM_MODEL` | `mlx-community/gemma-3-4b-it-4bit` | テキスト生成モデル |
| `VLM_MODEL` | `mlx-community/Phi-3.5-vision-instruct-4bit` | ビジョンモデル |
| `PORT` | `8000` | サーバーポート |

---

## パフォーマンス (Apple M3 Pro)

| ステージ | 所要時間 |
|----------|---------|
| 音声認識 (ASR) | ~1–3s |
| テキスト生成 (LLM) | ~1–3s（ツールなし）|
| 画像説明 (VLM) | ~10–15s（カメラ使用時） |
| 音声合成 (TTS) | ~1–3s |

---

## Acknowledgments

- [fikrikarim/parlor](https://github.com/fikrikarim/parlor) — 元のリポジトリ
- [Gemma 3](https://ai.google.dev/gemma) by Google DeepMind
- [mlx-lm](https://github.com/ml-explore/mlx-lm) by Apple
- [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) by Apple
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) by Blaizzy
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) TTS by Hexgrad
- [Silero VAD](https://github.com/snakers4/silero-vad)

## License

[Apache 2.0](LICENSE)
