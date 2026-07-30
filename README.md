# Friday

Friday is a self-hosted, voice-capable coding assistant harness that runs
entirely on local hardware — no cloud inference, no third-party API calls.
It's designed to be reachable two ways: a terminal over SSH, and a browser
UI, from any device on the local network, while the actual model inference
runs on a single dedicated GPU host.

## Goal

The end target is a voice-driven coding assistant loop:

1. **Speech-to-text** turns spoken input into a transcript.
2. **A routing model** decides which tool or function the request maps to.
3. **A large code-focused model** handles the actual reasoning and code
   generation, with the tools it needs to read/write files, run commands,
   and interact with a coding project.
4. **Text-to-speech** turns the response back into audio.
5. A **retrieval-augmented memory layer** lets the assistant recall
   relevant past conversation and code context beyond what fits in the
   model's context window.
6. **Voice-based speaker identification** lets the assistant recognize who
   it's talking to.

### Model stack and hardware split

Everything below is planned to run on a single RTX 3090 (24GB VRAM) host:

- A ~30B-parameter mixture-of-experts code model (quantized) for reasoning
  and code generation.
- A small dedicated tool-routing model that takes a message plus tool
  schemas and outputs which function to call — kept separate from the code
  model so tool dispatch stays fast and cheap.

CPU-side (no GPU required):

- A small embedding model for the RAG memory pipeline.
- Speech-to-text (faster-whisper).
- Text-to-speech (Piper).
- Speaker-embedding model for voice-based identification.

PostgreSQL + pgvector handles the vector store for RAG memory, disk/CPU
only.

## Current status

The project is at the "walking skeleton" stage: a complete, working voice
round trip — microphone in, transcription, a response, synthesized speech
out — proven across all three components before any real model
intelligence is wired in. The large code model and tool-routing model
described above are not yet integrated; the assistant currently echoes
back what it heard instead of reasoning about it, so the plumbing between
components can be verified independently of model quality.

### What's built

**Backend** (`backend/`) — a FastAPI service exposing a single WebSocket
session endpoint. It buffers incoming audio, runs real speech-to-text
(faster-whisper) and real text-to-speech (Piper) inference, and streams
the transcript, a response, and the synthesized audio back to the client
over the same connection. Model inference is offloaded to a thread pool so
one session's transcription doesn't stall others. Buffered audio is capped
per utterance to bound memory use.

**Native client** (`client/`) — a system-tray application for desktop use.
Hold a hotkey to record, release to send; the response prints to the
terminal and plays through the system's audio output. Handles dropped
connections and slow/unresponsive servers without hanging the input
listener.

**Browser client** (`frontend/`) — a framework-free web page served
directly by the backend. Supports mouse, touch, and keyboard (Space or
Enter) for push-to-talk, with sequential audio playback scheduling so
streamed response audio doesn't overlap or garble.

Every component has its own test suite (33 automated tests across backend
and client) plus manual verification checkpoints for the parts that need
real audio hardware and a live model to confirm — automated tests mock the
model layer, so end-to-end audio correctness is checked by hand at each
integration point.

## Project layout

```
backend/    FastAPI server, speech-to-text, text-to-speech, WebSocket session handling
client/     Native system-tray push-to-talk client
frontend/   Browser-based push-to-talk client
```

## Running it

Each of `backend/` and `client/` is an independent Python project managed
with [uv](https://github.com/astral-sh/uv).

```bash
# Backend
cd backend
uv sync
uv run python -m app.main
```

By default the backend listens on `127.0.0.1:8000`. To accept connections
from other machines on the network, set `FRIDAY_HOST=0.0.0.0` before
starting it.

```bash
# Native client (separate terminal)
cd client
uv sync
uv run python app.py
```

For the browser client, once the backend is running, open
`http://localhost:8000/` (or the backend host's address) in a browser.

### Tests

```bash
cd backend && uv run pytest
cd client && uv run pytest
```
