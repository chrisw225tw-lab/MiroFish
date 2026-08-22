# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MiroFish is a multi-agent AI prediction engine. Users upload seed documents (PDF/MD/TXT), which are processed into a Zep-backed knowledge graph. The system then generates agent personalities from the ontology, runs a multi-agent simulation (via the OASIS framework), and produces a report with interactive chat.

## Commands

### Development (local, no Docker)
```bash
npm run setup:all     # Install all Node + Python (uv) dependencies
npm run dev           # Start both frontend (port 3000) and backend (port 5001)
npm run backend       # Backend only: cd backend && uv run python run.py
npm run frontend      # Frontend only: cd frontend && npm run dev
npm run build         # Build frontend for production
```

### Docker
```bash
docker compose up -d      # Run with Docker (requires .env)
docker compose logs -f    # Follow logs
```

### Backend tests (pytest available but not wired to npm scripts)
```bash
cd backend && uv run pytest
```

## Architecture

### Workflow (left to right)
```
Upload → Graph Build → Ontology Generation → Profile Generation → Simulation → Report → Interactive Chat
```
Each step is a separate API call. The frontend drives through this flow using `project_id` as the persistent identifier. Long-running steps return a `task_id` that the frontend polls.

### Backend (`backend/app/`)
- **`api/`** — Flask blueprints: `graph.py`, `simulation.py`, `report.py`. The simulation blueprint is the largest (~3000 lines), handling the full simulation lifecycle.
- **`services/`** — All business logic:
  - `graph_builder.py` — Sends chunked text to Zep Cloud to build a knowledge graph (nodes = entities, edges = relationships).
  - `ontology_generator.py` — LLM call to extract typed entities/relationships from the graph.
  - `oasis_profile_generator.py` — LLM call to generate detailed agent personality profiles.
  - `simulation_config_generator.py` — Maps profiles to OASIS agent configs.
  - `simulation_runner.py` — Spawns an OASIS simulation subprocess; handles Twitter/Reddit platform modes.
  - `simulation_ipc.py` — IPC queues for passing agent decisions/actions between processes.
  - `simulation_manager.py` — Orchestrates the above services in sequence.
  - `report_agent.py` — ReflectionAgent pattern; uses Zep tools to retrieve memory and generate structured reports.
  - `zep_tools.py`, `zep_entity_reader.py`, `zep_graph_memory_updater.py` — Wrappers for Zep Cloud API.
- **`models/`** — `ProjectManager` (in-memory + file persistence) and `TaskManager` (background task tracking).
- **`utils/llm_client.py`** — OpenAI-compatible LLM wrapper. Key methods: `chat()` and `chat_json()`. Default `max_tokens=4096`.

### Frontend (`frontend/src/`)
- **`views/MainView.vue`** — Stepper orchestrator. Hosts Step1–Step5 components.
- **`components/`** — One component per workflow step (`Step1GraphBuild.vue` → `Step5Interaction.vue`).
- **`api/`** — Three modules (`graph.js`, `simulation.js`, `report.js`) wrapping Axios calls to the Flask backend.
- **`store/pendingUpload.js`** — Shared state for files pending upload.
- D3.js is used for graph visualization.

### External Services
- **Graph backend** — selected by `ZEP_BACKEND`:
  - `cloud` (default) — **Zep Cloud** (`ZEP_API_KEY`), upstream behaviour.
  - `graphiti` — self-hosted **graphiti-core + FalkorDB**, see `docs/SELFHOST-GRAPHITI.md`.
    Implemented in `backend/app/services/graphiti_backend/`, which mimics the Zep SDK's
    `.graph` / `.batch` namespaces so no call site changes between backends.
- **LLM Proxy** — OpenAI-compatible endpoint (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL_NAME`). Currently configured to `cli-proxy-api` over Tailscale at `http://100.90.255.68:8317/v1`.
- Optional boost LLM: `LLM_BOOST_*` env vars for a second, faster model.
- Optional model failover: `LLM_MODEL_FALLBACKS` (comma-separated, same base URL/key).
  5xx/429/timeout/connection errors switch to the next model and park the failed one
  for 60s; 4xx propagates. Currently `Kimi-k2.6` → `gpt-5.4-mini`.

### LLM Proxy (`cli-proxy-api`)

This project does **not** call any LLM provider directly. All LLM calls go through CLIProxyAPI at `http://100.90.255.68:8317/v1`, configured in `/home/chris/cliproxyapi/config.yaml`.

The active runtime models are:

| Model name | Backend | Auth source |
|---|---|---|
| `Kimi-k2.6` | Kimi coding route | CLIProxyAPI `claude-api-key` compatibility entry |
| `gpt-5.4-mini` | OpenAI/Codex | CLIProxyAPI authenticated OpenAI/Codex account |

## Key Implementation Details

### LLM Client (`backend/app/utils/llm_client.py`)
- `kimi-for-coding` and similar reasoning models put their answer in `content` and thinking in `reasoning_content`. They need substantial `max_tokens` (2000+) to finish reasoning before outputting `content` — with too few tokens, `content` will be an empty string.
- `content` can be `None` from some models; always guard with `or ""`.
- `chat_json()` passes `response_format={"type": "json_object"}` and strips markdown code fences.

### Simulations run as subprocesses
`simulation_runner.py` spawns OASIS in a child process. Flask registers a shutdown handler to clean up on exit. The `simulation_ipc.py` module bridges the subprocess back to the API layer via queues.

### Project persistence
Projects are stored in memory (Python dict) and serialized to `backend/uploads/<project_id>/`. Restarting the backend loses in-memory state unless loaded from disk.

### File uploads
Accepted: `.pdf`, `.md`, `.txt`, `.markdown`. Max 50 MB. Stored under `backend/uploads/`.

### Encoding
`run.py` forces UTF-8 stdout/stderr on Windows. Flask JSON is configured with `ensure_ascii=False` to pass Chinese text through unescaped.
