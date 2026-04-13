# Parlor (日本語対応・MLX版)

オンデバイスでリアルタイムに動く、マルチモーダルAIとの音声・カメラ会話システム。
**Apple Silicon Mac** 上で全モデルをローカル実行します。

このリポジトリは [fikrikarim/parlor](https://github.com/fikrikarim/parlor) をフォークし、
**日本語対応** と **MLXバックエンドへの切替** を中心に改変したものです。

---

## 開発の経緯（裏話）

もともと **Gemma 4** の学習用として始めたリポジトリでした。

Gemma 4 は Google が2026年4月にリリースした最新マルチモーダルモデルで、
E2B・E4B などエッジデバイス向けの小型モデルが存在します。
元のParlor はこれを **LiteRT-LM**（サーバーサイド）で動かす構成になっていました。

実際に動かしてみると、LiteRT-LM 自体は問題なく動いたのですが、
Apple Silicon Mac では **GPU sampler が非対応**となって CPU フォールバックで動作し、
応答に **60秒以上** かかる状態になりました。

「もっと速くしたい」という動機から、Apple Silicon に最適化された
**MLXバックエンド（mlx-vlm）** への切り替えを選びました。

ところが Gemma 4 リリース直後は mlx-lm が **Gemma 4 に未対応**
（mlx-vlm は Day-0 対応、mlx-lm は 4月8日の v0.31.2 で対応）だったため、
一時的に **Gemma 3 4B 4-bit**（mlx-lm 経由）で動かしていました。

その後、mlx-vlm が Gemma 4 の ASR・ビジョン・テキスト生成を 1 モデルで担えることが確認できたため、
**Gemma 4 E2B に統一**し、Whisper・Gemma 3・Phi-3.5-vision の 3 モデルを廃して
シンプルな構成に整理しました。

---

## 元のParlor からの主な差分

| 項目 | 元のParlor | このフォーク |
|------|-----------|-------------|
| 推論バックエンド | LiteRT-LM (Google) | MLX (Apple Silicon) |
| 言語モデル | Gemma 4 E2B | Gemma 4 E2B |
| 音声認識 (ASR) | LiteRT内蔵 (Gemma 4) | Gemma 4 E2B 内蔵 ASR (mlx-vlm) |
| TTS言語 | 英語 (af_heart) | 日本語 (jf_alpha) |
| フォネマイザ | misaki[en] | misaki[en,ja] + pyopenjtalk |
| ツール呼び出し | Gemma 4 ネイティブ function calling | JSON パース + サーバー側実行ループ |
| ビジョン | Gemma 4 マルチモーダル | Gemma 4 E2B (mlx-vlm) |

### アーキテクチャ

```
Browser (マイク + カメラ)
    │
    │  WebSocket (WAV音声 + JPEG フレーム)
    ▼
FastAPI サーバー
    ├── Gemma 4 E2B via mlx-vlm   →  音声認識 (ASR) + テキスト生成 + カメラ映像理解
    └── Kokoro TTS (mlx-audio / jf_alpha)   →  日本語音声合成
    │
    │  WebSocket (ストリーミング音声チャンク)
    ▼
Browser (再生 + 文字起こし表示)
```

### ツール呼び出しの仕組み

mlx-vlm 経由の Gemma 4 はネイティブ function calling をサポートしていないため、
LLM に JSON フォーマットで出力させ、サーバー側でパースして実行するループを実装しています。

```
ユーザー: 「今何時？」
  ↓
Gemma 4 E2B: {"tool_call": {"name": "get_current_time", "arguments": {}}}
  ↓
サーバー: datetime.now() を実行 → "2026年04月11日 01時47分"
  ↓
Gemma 4 E2B: "今は2026年4月11日の1時47分ですよ！"
  ↓
Kokoro TTS: 音声を生成して返す
```

### 音声認識と画像理解

Gemma 4 E2B は音声・画像・テキストを 1 モデルで処理します。
音声は WAV を一時ファイル経由で渡し、画像は PIL Image として直接渡します。

```
音声 (WAV) + テキスト [+ 画像]
  ↓
Gemma 4 E2B (mlx-vlm)
  ↓
テキスト応答
```

---

## セットアップ

### 必要環境

- Python 3.12
- macOS + Apple Silicon (M1/M2/M3)
- 空きメモリ: ~3GB（Gemma 4 E2B ~2.3GB + Kokoro TTS）

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

初回起動時にモデルが自動ダウンロードされます（合計 ~2GB）。

---

## 環境変数

| 変数名 | デフォルト | 説明 |
|--------|-----------|------|
| `VLM_MODEL` | `mlx-community/gemma-4-E2B-it-4bit` | Gemma 4 E2B モデル（ASR・LLM・VLM を兼ねる） |
| `PORT` | `8000` | サーバーポート |

---

## パフォーマンス

| ステージ | 所要時間 |
|----------|---------|
| 音声認識 (ASR) | ~2–5s |
| テキスト生成 (LLM) | ~1–3s（ツールなし）|
| 音声合成 (TTS) | ~1–3s |

---

## Acknowledgments

- [fikrikarim/parlor](https://github.com/fikrikarim/parlor) — 元のリポジトリ
- [Gemma 4](https://ai.google.dev/gemma) by Google DeepMind
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) by Blaizzy
- [mlx-audio](https://github.com/ml-explore/mlx-audio) by Apple
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) TTS by Hexgrad
- [Silero VAD](https://github.com/snakers4/silero-vad)

## License

[Apache 2.0](LICENSE)
