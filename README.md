# Midstone 4: local Ollama LLM client

This project uses only Ollama's local OpenAI-compatible endpoint. It does not have a cloud-provider fallback. The SDK's `LLM_API_KEY=ollama` value is a local placeholder, not a credential.

## Setup

1. Install Ollama from [ollama.com/download](https://ollama.com/download) if it is not already installed.
2. Start Ollama (`ollama serve` if it is not running).
3. Download the configured model once: `ollama pull qwen2.5:7b`.
4. Copy `.env.example` to `.env` and adjust `DATABASE_URL` if your Midstone 2 database is elsewhere. `.env` is ignored by Git.
5. Install dependencies with `python -m pip install -r requirements.txt`.
6. Run tests with `python -m pytest -v`.
7. Run the three-turn demo with `python chat_demo.py`. It appends execution events to `run_log.jsonl` only after live requests succeed.

To check the model manually, run `ollama run qwen2.5:7b`. The OpenAI-compatible endpoint is `http://localhost:11434/v1`.

The demo uses the actual `Midstone 2/kaveri.db` next to this project when present, or the `DATABASE_URL` from `.env`. It runs a single parameterized read-only revenue query. It does not create or seed a database.
