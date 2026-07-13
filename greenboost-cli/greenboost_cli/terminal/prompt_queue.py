"""
Prompt queue — buffers user input while the model is processing.

Users can type new prompts while a response is being generated; those
inputs land here instead of being dropped. The queue is drained
automatically (one item at a time) as soon as the model finishes.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QueuedPrompt:
    id: int
    text: str


class PromptQueue:
    """Thread-safe FIFO queue for pending user prompts."""

    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self._items:  list[QueuedPrompt] = []
        self._counter = 0

    # ── Producers ──────────────────────────────────────────────────────────

    def enqueue(self, text: str) -> QueuedPrompt:
        with self._lock:
            self._counter += 1
            item = QueuedPrompt(id=self._counter, text=text)
            self._items.append(item)
            return item

    # ── Consumers ──────────────────────────────────────────────────────────

    def dequeue(self) -> Optional[QueuedPrompt]:
        with self._lock:
            return self._items.pop(0) if self._items else None

    # ── Management ─────────────────────────────────────────────────────────

    def delete(self, index: int) -> bool:
        """Delete by 1-based position in the queue. Returns True on success."""
        with self._lock:
            idx = index - 1
            if 0 <= idx < len(self._items):
                self._items.pop(idx)
                return True
            return False

    def edit(self, index: int, new_text: str) -> bool:
        """Replace text at 1-based position. Returns True on success."""
        with self._lock:
            idx = index - 1
            if 0 <= idx < len(self._items):
                self._items[idx].text = new_text
                return True
            return False

    def clear(self) -> int:
        """Clear all queued items. Returns the number cleared."""
        with self._lock:
            count = len(self._items)
            self._items.clear()
            return count

    def snapshot(self) -> list[QueuedPrompt]:
        """Return a point-in-time copy of the queue."""
        with self._lock:
            return list(self._items)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def __bool__(self) -> bool:
        return len(self) > 0
