"""
app/ui/server.py

FastAPI server that serves the dashboard and relays pipeline events
to connected browser clients over WebSocket.

Runs in a daemon thread alongside the main pipeline loop.
Does not import from app.core.loop — decoupled via event_bus.

Endpoints
---------
GET  /          → dashboard HTML
WS   /ws        → real-time event stream (JSON messages)
GET  /health    → {"status": "ok"}
"""

import asyncio
import logging
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.ui.event_bus import event_bus

logger = logging.getLogger(__name__)

app = FastAPI(title="Eyes and Ears", docs_url=None, redoc_url=None)

# Serve static files (index.html lives here)
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def dashboard():
    index = _static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"error": "index.html not found"}, status_code=404)


@app.get("/health")
async def health():
    return {"status": "ok", "clients": event_bus.client_count}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    client_q = event_bus.subscribe()
    logger.info("WebSocket client connected | total=%d", event_bus.client_count)

    try:
        while True:
            # Drain any pending events
            drained = 0
            while drained < 20:   # max 20 messages per iteration
                try:
                    msg = client_q.get_nowait()
                    await ws.send_json(msg)
                    drained += 1
                except queue.Empty:
                    break

            # Keep-alive: receive (handles disconnect detection)
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=0.05)
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break

    except Exception as exc:
        logger.debug("WebSocket error: %s", exc)
    finally:
        event_bus.unsubscribe(client_q)
        logger.info(
            "WebSocket client disconnected | remaining=%d", event_bus.client_count
        )


# ---------------------------------------------------------------------------
# Server lifecycle — call from loop.py
# ---------------------------------------------------------------------------

_server_thread: threading.Thread | None = None


def start_ui_server(host: str = "0.0.0.0", port: int = 7860) -> None:
    """
    Start the UI server in a background daemon thread.
    Returns immediately. Safe to call once.
    """
    global _server_thread

    if _server_thread and _server_thread.is_alive():
        logger.warning("UI server already running")
        return

    def _run():
        import uvicorn
        uvicorn.run(
            app,
            host      = host,
            port      = port,
            log_level = "warning",
        )

    _server_thread = threading.Thread(target=_run, name="ui-server", daemon=True)
    _server_thread.start()
    logger.info("UI server started | http://%s:%d", host if host != "0.0.0.0" else "localhost", port)
