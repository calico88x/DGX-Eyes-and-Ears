"""
app/ui/event_bus.py

Thread-safe publish/subscribe bus.

The pipeline loop publishes events here from any thread.
The WebSocket server subscribes and forwards to connected browser clients.

Pattern: one Queue per connected client, populated by publish().
The WebSocket handler polls its queue asynchronously.

No imports from app.core or app.services — keeps this dependency-free.
"""

import queue
import threading
from typing import Any

__all__ = ["event_bus"]


class EventBus:
    """
    Broadcast events from the pipeline loop to all connected WebSocket clients.

    Usage
    -----
    # In loop.py (any thread):
    from app.ui.event_bus import event_bus
    event_bus.publish({"type": "state_change", "state": "LISTENING", ...})

    # In server.py (WebSocket handler):
    q = event_bus.subscribe()
    try:
        msg = q.get_nowait()
    except queue.Empty:
        await asyncio.sleep(0.05)
    finally:
        event_bus.unsubscribe(q)
    """

    def __init__(self):
        self._lock:    threading.Lock       = threading.Lock()
        self._clients: list[queue.Queue]    = []

    def subscribe(self) -> queue.Queue:
        """Register a new subscriber. Returns its dedicated queue."""
        q: queue.Queue = queue.Queue(maxsize=200)
        with self._lock:
            self._clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        """Remove a subscriber's queue (call on WebSocket disconnect)."""
        with self._lock:
            try:
                self._clients.remove(q)
            except ValueError:
                pass

    def publish(self, event: dict[str, Any]) -> None:
        """
        Broadcast an event to all connected clients.
        Non-blocking — drops the event for any client whose queue is full.
        Safe to call from any thread.
        """
        with self._lock:
            clients = list(self._clients)

        for q in clients:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass   # slow client — drop rather than block

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)


# Module-level singleton — import this everywhere
event_bus = EventBus()
