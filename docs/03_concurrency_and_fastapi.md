# Module 03: Operating System Concurrency, Asyncio & FastAPI Enterprise Architecture

## 1. Operating System & Python Concurrency Primitives

### 1.1 The Global Interpreter Lock (GIL) & Workload Characterization

CPython's memory management relies on reference counting, protected by the **Global Interpreter Lock (GIL)**. The GIL prevents multiple native OS threads from executing Python bytecode simultaneously within a single process. Understanding the boundaries of the GIL is essential for architecting enterprise backends:

```text
[ CPU-Bound Workload ]        VS.       [ I/O-Bound Workload (Tech Radar) ]
+-------------------------+             +-------------------------+
| Native Thread 1: COMPUTE|             | Coroutine 1: Await DB   | ---> OS Socket Wait
| Native Thread 2: BLOCKED|             | Coroutine 2: Await LLM  | ---> Network I/O Wait
+-------------------------+             | Coroutine 3: Await HTTP | ---> Disk/API Wait
Thread contention & lock thrashing      +-------------------------+
Requires multiprocessing                Single thread scales to 10,000+ tasks
```

#### Why `asyncio` is Optimal for LLM & Webhook Workloads
The Tech Radar Agent's workload is overwhelmingly **I/O-bound**:
- Polling external APIs (arXiv, HackerNews, GitHub)
- Asynchronous database queries across PostgreSQL connection pools
- Waiting for LLM completions from Groq LPUs (ultra-low latency streaming inference)
- Dispatching webhooks to Discord and Telegram Bot API

In an I/O-bound paradigm, preemptive multithreading introduces unnecessary context switching overhead, stack memory allocations (~8MB per thread), and locking complexity. `asyncio` employs cooperative multitasking on a single OS thread, multiplexing thousands of concurrent connections using kernel-level event notification facilities (**epoll** on Linux, **kqueue** on macOS).

### 1.2 Event Loop Mechanics & Non-Blocking HTTP (`httpx`)

The core engine is the `asyncio` event loop. When a coroutine executes:

```python
async with httpx.AsyncClient() as client:
    response = await client.post(endpoint, json=payload)
```

1. The coroutine reaches the `await` expression and yields control back to the event loop.
2. The HTTP client registers the underlying TCP socket descriptor with the OS kernel's `epoll` listener (`EPOLLOUT` for writing, `EPOLLIN` for reading).
3. The event loop pauses the coroutine frame in memory and immediately executes other ready tasks (e.g. processing a Discord button click or evaluating another release's regex rules).
4. When the external server responds, the Linux kernel wakes the `epoll_wait` syscall, and the event loop resumes the suspended coroutine from its exact execution frame.

Zero CPU cycles are wasted while waiting for network packet round-trips.

### 1.3 Subprocess & Thread Offloading for WeasyPrint PDF Rendering

While ingestion and API dispatch are I/O-bound, **WeasyPrint PDF compilation is intensely CPU-bound and memory-bound**:
- Parsing HTML/CSS document trees
- Computing paged media layouts, page breaks, and line wraps
- Invoking C libraries (Cairo, Pango, HarfBuzz, Fontconfig) for glyph rendering and font rasterization

If WeasyPrint were invoked directly within an async endpoint:
```python
# ANTI-PATTERN: Freezes the entire FastAPI process!
weasyprint.HTML(string=html).write_pdf("out.pdf")
```
The CPU-intensive rendering loop would monopolize the single Python thread, blocking the `asyncio` event loop. All incoming webhooks, Discord button clicks, and health checks would hang until rendering completed, violating enterprise latency SLAs.

#### Enterprise Solution: `asyncio.to_thread` Worker Pool
We offload WeasyPrint rendering to a managed thread pool executor:

```python
await asyncio.to_thread(self._sync_render_pdf, html_content, output_path)
```

`asyncio.to_thread` dispatches the blocking C-extension call to Python's internal `ThreadPoolExecutor`. The event loop continues processing asynchronous coroutines without interruption. When the worker thread finishes writing the PDF artifact to disk, it signals the event loop to resume the caller.

---

## 2. FastAPI Enterprise Architecture & Microservices Design

### 2.1 Dependency Injection for Database Sessions & Client Lifecycles

FastAPI's Dependency Injection (`Depends`) framework enforces clean inversion of control, transaction boundaries, and resource isolation:

```python
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

#### Key Enterprise Guarantees:
1. **Per-Request Transaction Boundaries:** Every incoming API request receives an isolated `AsyncSession`. If an exception occurs, the generator automatically triggers `session.rollback()`.
2. **Leak Prevention:** The `finally` block guarantees that database connections are returned to the `asyncpg` pool, even during client disconnections or timeouts.
3. **Mockability for Testing:** In test suites (`tests/`), dependencies are overridden cleanly:
   ```python
   app.dependency_overrides[get_db] = override_test_db
   ```

### 2.2 Pydantic v2 Schemas & High-Performance Validation

The codebase leverages **Pydantic v2**, written in Rust (`pydantic-core`), providing a 5x to 20x performance improvement over Pydantic v1:

- **Strict Type Validation:** Input schemas reject malformed inputs before reaching business logic (`model_config = ConfigDict(extra="forbid")`).
- **Mathematical Bounds Enforcement:**
  ```python
  similarity_threshold: Optional[float] = Field(default=0.82, ge=0.0, le=1.0)
  ```
- **Serialization Safety:** Database entities are converted to client-facing DTOs via `ConfigDict(from_attributes=True)`, preventing unintended leakage of internal schema fields or sensitive tokens.

### 2.3 Non-Blocking Periodic Ingestion & Lifespan Architecture

Background polling is decoupled from HTTP request lifecycles using FastAPI's modern `lifespan` context manager:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP:
    await init_db()
    _ = get_radar_graph()
    polling_task = asyncio.create_task(periodic_ingestion_worker())
    
    yield
    
    # SHUTDOWN:
    polling_task.cancel()
    await polling_task
    await close_db()
```

#### Operational Advantages:
1. **Deterministic Startup Order:** Database tables, HNSW indexes, and LangGraph checkpointer tables are guaranteed to be initialized before the server accepts incoming traffic.
2. **Graceful Teardown:** When Kubernetes or Docker transmits a `SIGTERM` signal, the lifespan manager cancels the background polling task, waits for in-flight transactions to settle, and disposes of the connection pool cleanly without dropping active database locks.
