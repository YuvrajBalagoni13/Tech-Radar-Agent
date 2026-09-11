# Autonomous Tech Radar & Just-In-Time Learning Agent

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20+%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![WeasyPrint](https://img.shields.io/badge/WeasyPrint-PDF%20Engine-brightgreen.svg)](https://weasyprint.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Enterprise Flagship Portfolio Project** targeting Elite Technology Engineer roles at premier technology consultancies and hyperscalers (such as Accenture Level 11, Staff AI Engineer, and Lead Systems Architect).

---

## 1. System Overview & Value Proposition

Modern software engineering teams suffer from severe information overload: hundreds of foundational LLM releases, framework updates, quantization kernels, and architectural preprints drop weekly. Unfiltered feeds create distraction and notification fatigue.

The **Autonomous Tech Radar & Just-In-Time Learning Agent** functions as an autonomous, self-tuning research copilot:
1. **Targeted Ingestion:** Periodically polls technical feeds (HackerNews API, arXiv AI papers, GitHub Trending).
2. **Three-Stage Triage Funnel:**
   - *Stage 1 (Deduplication):* Sub-millisecond SHA-256 hash matching against PostgreSQL `processed_releases`.
   - *Stage 2 (Deterministic Negative Match):* Regex pattern matching on developer-ignored keywords (e.g. `crypto`, `NFT`) with **$0 LLM token cost**.
   - *Stage 3 (Dense Vector Similarity):* 1536-dimensional cosine distance evaluation against dynamic user profiles stored in **pgvector** using **HNSW indexing**.
3. **Interactive Multi-Channel Dispatch:** Posts rich embeds with interactive action buttons (`[ 📚 Generate Hands-On Tutorial ]` and `[ ⏭️ Skip / Ignore ]`) to **Discord**.
4. **Stateful Checkpointing & Escalation:** Persists thread state into PostgreSQL using LangGraph's `AsyncPostgresSaver`. If an alert is unacknowledged after a configurable SLA (e.g., 6 hours), an escalation node alerts the developer on **Telegram** (100% free via Telegram Bot API with inline action buttons).
5. **Domain-Scoped Deep Research:** On developer approval, executes constrained technical search queries scoped to the user's declared tech stack.
6. **Publication-Grade PDF Synthesis:** Compiles an executive summary, architectural blueprint, Hype vs. Reality trade-off matrix, and runnable Python code into an executive whitepaper via **WeasyPrint** and delivers it directly to Discord and Telegram.

---

## 2. Architectural Blueprint

```text
+-----------------------+      1. Ingest Payload       +-------------------------+
| External Event Stream | ---------------------------> | Async Ingestion Gateway |
+-----------------------+                              +-------------------------+
                                                                    |
                                                       2. Verify Hash / SHA-256
                                                                    v
+-----------------------+      3. Cosine Evaluation    +-------------------------+
| PostgreSQL + pgvector | <--------------------------- | Deterministic Triage    |
+-----------------------+                              +-------------------------+
           |                                                        |
           | 4. Return Top-K                                        | 5. Dispatch
           v                                                        v
+-----------------------+                              +-------------------------+
| LangGraph Checkpoint  | <--------------------------- | Human-in-the-Loop Node  |
+-----------------------+    Resume on Callback        +-------------------------+
```

---

## 3. Directory Layout

```text
tech-radar-agent/
├── docker-compose.yml                  # PostgreSQL 16 (pgvector), Redis, and FastAPI App
├── Dockerfile                          # Multi-stage production container with WeasyPrint & CFFI
├── requirements.txt                    # Validated pinned enterprise dependencies
├── .env.example                        # Template environment configurations
├── pytest.ini                          # Pytest configuration
├── README.md                           # Master architectural documentation
├── app/
│   ├── __init__.py
│   ├── main.py                         # FastAPI server, Discord interaction callbacks, Telegram webhook
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                   # Pydantic v2 Settings (DB URI, API tokens, thresholds, timeouts)
│   │   └── database.py                 # Async SQLAlchemy engine, asyncpg pool, pgvector initialization
│   ├── bot/
│   │   ├── __init__.py
│   │   ├── discord_client.py           # Discord bot instance, slash commands (/radar track, /radar ignore)
│   │   └── telegram_handler.py         # Telegram webhook parsing, inline buttons & command processor
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py                  # Pydantic v2 schemas for payloads, DTOs, API inputs/outputs
│   │   └── entities.py                 # SQLAlchemy entities (UserProfile, ProcessedRelease with HNSW)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ingestion.py                # Ingestion feeds (HackerNews, arXiv, GitHub) with SHA-256 deduplication
│   │   ├── notifier.py                 # OOP Strategy Pattern: BaseNotifier, DiscordNotifier, TelegramNotifier
│   │   ├── search.py                   # Domain-scoped Tavily / search scraper & markdown parser
│   │   └── pdf_generator.py            # WeasyPrint PDF compiler with custom CSS styling and syntax formatting
│   └── agent/
│       ├── __init__.py
│       ├── state.py                    # LangGraph AgentState TypedDict
│       ├── graph.py                    # StateGraph definition, conditional edges, AsyncPostgresSaver setup
│       └── nodes/
│           ├── __init__.py
│           ├── domain_filter_node.py   # Multi-stage deterministic & pgvector cosine filter
│           ├── notification_node.py    # Discord embed dispatcher & state checkpoint setup
│           ├── escalation_node.py      # Telegram timeout escalation sender
│           ├── research_node.py        # Domain-scoped multi-query retrieval engine
│           ├── synthesis_node.py       # Hype vs. Reality and hands-on tutorial drafting
│           └── compiler_node.py        # PDF compilation and delivery node
├── tests/
│   ├── __init__.py
│   ├── test_filter_node.py             # Deduplication, negative keywords, and cosine similarity tests
│   ├── test_notifier_strategy.py       # OOP Strategy pattern and notification dispatch tests
│   └── test_graph_flow.py              # End-to-end StateGraph transitions and checkpoint resumption tests
└── docs/                               # 4 In-Depth Technical Manuals (Mandatory)
    ├── 01_architecture_and_langgraph.md
    ├── 02_pgvector_and_dbms_internals.md
    ├── 03_concurrency_and_fastapi.md
    └── 04_interview_defense_cheatsheet.md
```

---

## 4. Quickstart with Docker Compose

### Prerequisites
- Docker Engine 24.0+ and Docker Compose v2
- Groq API Key (`gsk_...`, or runs in deterministic offline mock mode)
- Discord Bot Token & Telegram Bot Token (100% free via @BotFather)

### Step 1: Clone and Configure Environment
```bash
cp .env.example .env
# Edit .env with your credentials
```

### Step 2: Spin Up Infrastructure
```bash
docker compose up --build -d
```
This launches:
- `techradar_postgres`: PostgreSQL 16 with `pgvector` extension and persistent storage volume
- `techradar_redis`: Redis 7 Alpine cache
- `techradar_app`: FastAPI asynchronous backend with WeasyPrint C-libraries pre-compiled

### Step 3: Verify Health Check
```bash
curl http://localhost:8000/health
```
Response:
```json
{
  "status": "healthy",
  "service": "tech-radar-agent",
  "environment": "production",
  "default_threshold": 0.82,
  "escalation_sla_hours": 6
}
```

---

## 5. User Interaction & Commands

### Channel Selection (`PRIMARY_NOTIFICATION_CHANNEL`)
You can choose which platform receives initial radar alerts:
- `PRIMARY_NOTIFICATION_CHANNEL=TELEGRAM` — Alerts go directly to Telegram (100% free, no Discord needed).
- `PRIMARY_NOTIFICATION_CHANNEL=DISCORD` — Alerts go to Discord first, and escalate to Telegram on SLA timeout.
- `PRIMARY_NOTIFICATION_CHANNEL=BOTH` — Alerts are dispatched to Discord and Telegram simultaneously.

### Discord Bot Direct Messages (DMs) & Slash Commands
The bot operates in **User Direct Message (DM) Mode** (by setting `DISCORD_USER_ID`) or Server Channel Mode:
- **Private DM Alerts:** The bot opens a private DM channel with your Discord user account and delivers radar embeds with interactive buttons (`[ 📚 Generate Hands-On Tutorial ]`, `[ ⏭️ Skip / Ignore ]`) and compiled PDF whitepapers directly to your DMs.
- **Direct Messaging the Bot:** You can message the bot directly in your DMs:
  - `!track <domain>` or `/radar track` - Appends technical domains and synchronizes your 1536-dim profile vector.
  - `!ignore <keywords>` or `/radar ignore` - Appends negative keywords (zero LLM token cost rejection).
  - `!threshold <0.0-1.0>` or `/radar threshold` - Adjusts cosine similarity cutoff.
  - `!status` or `/radar list` - Displays active profile, tracked domains, and vector sync state.
  - `!yes <thread_id> [prompt]` - Resumes interrupted checkpoint with custom instructions.
  - `!skip <thread_id>` - Dismisses release item.
  - `!help` - Displays interactive guidance in DMs.

### Telegram Bot Commands & Inline Buttons (100% Free)
- **Inline Keyboard:** Every alert arrives with 1-click interactive buttons:
  - `[ 📚 Generate Hands-On Tutorial ]` -> Automatically resumes LangGraph to research, draft, and compile PDF
  - `[ ⏭️ Skip / Ignore ]` -> Dismisses the release and marks state as `USER_IGNORED`
- `/track <domain or comma-separated keywords>`  
  *Appends technical domains to your developer profile and updates your 1536-dim vector.*
- `/ignore <keywords>`  
  *Appends negative filter keywords (zero LLM token cost rejection).*
- `/threshold <value>`  
  *Sets similarity threshold between 0.0 and 1.0 (e.g. `/threshold 0.85`).*
- `/status`  
  *Displays your active tracked domains, negative keywords, and vector sync state.*
- `/yes <thread_id> [optional custom instructions]`  
  *Resumes interrupted LangGraph checkpoint with custom developer guidance.*
- `/skip <thread_id>`  
  *Dismisses interrupted release item.*

---

## 6. Running Automated Tests

Run the complete test suite using the isolated runner script:
```bash
./test.sh
```

All 13 unit and integration tests validate:
- Three-stage filtering pipeline (SHA-256 deduplication, negative regex, pgvector cosine calculation)
- OOP Strategy pattern, Discord rich embed builders, and Telegram Bot dispatchers
- End-to-end LangGraph StateMachine compilation, interrupt checkpoints, and thread resumption

---

## 7. Deep Technical Documentation

Comprehensive architectural and systems manuals are located in `/docs`:
- [01. Systems Architecture & LangGraph Internals](docs/01_architecture_and_langgraph.md): Event-driven sequence diagrams, state immutability, `AsyncPostgresSaver` mechanics, and OOP design patterns.
- [02. PostgreSQL, pgvector & DBMS Internals](docs/02_pgvector_and_dbms_internals.md): Complete DDL, HNSW vs. IVFFlat comparison, vector mathematics, and ACID guarantees.
- [03. Concurrency, Asyncio & FastAPI Architecture](docs/03_concurrency_and_fastapi.md): CPython GIL analysis, event loop mechanics, WeasyPrint thread offloading, and dependency injection.
- [04. Elite Technology Engineer Interview Defense Cheatsheet](docs/04_interview_defense_cheatsheet.md): 10 rigorous systems interview questions and model answers targeting Accenture Level 11 and Staff Architect roles.
