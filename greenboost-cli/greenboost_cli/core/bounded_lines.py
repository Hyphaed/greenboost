"""
Bounded line decoder for subprocess/stream output without unbounded buffering.

Ported from NemoClaw's core/bounded-line-transcript.ts (Apache-2.0, design reference).
A misbehaving subprocess or stream that emits megabytes of data without a line break
would grow an internal buffer without bound in naive readline() loops. This decoder
enforces a maximum pending buffer size; when exceeded, it keeps only the tail and
tracks how many characters were dropped, so truncation is visible (not silent).
"""
from __future__ import annotations

import codecs
from typing import Callable


class BoundedLineDecoder:
    """
    Decodes subprocess/stream output into complete lines without letting
    a chunk that never emits a line break grow an unbounded pending buffer.

    Incremental UTF-8 decoder across chunk boundaries. When a line terminator
    (\\n, \\r\\n, or bare \\r) is found, everything up to it is flushed as one
    complete line via the on_line callback. If a line's accumulated length
    without seeing a line terminator exceeds max_pending_chars, the decoder
    keeps only the LAST max_pending_chars of it and prepends a marker to
    indicate how many characters were omitted—so truncation is visible, never
    silent.

    Args:
        max_pending_chars: Maximum characters allowed in the pending buffer
            without a line terminator before truncation occurs. Must be a
            positive integer.
        on_line: Callback invoked once per complete line (including unterminated
            final lines at end()). Receives the flushed line as a str.
            If truncation occurred, the line is prefixed with a marker showing
            the count of omitted characters.

    Raises:
        ValueError: If max_pending_chars is not a positive integer, or if
            write() is called after end().
    """

    def __init__(self, max_pending_chars: int, on_line: Callable[[str], None]) -> None:
        if not isinstance(max_pending_chars, int) or max_pending_chars < 1:
            raise ValueError("max_pending_chars must be a positive integer")
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._max_pending_chars = max_pending_chars
        self._on_line = on_line
        self._pending = ""
        self._omitted_pending_chars = 0
        self._skip_leading_linefeed = False
        self._ended = False

    def write(self, chunk: bytes) -> None:
        """
        Feed raw bytes to the decoder.

        Args:
            chunk: Raw bytes from a subprocess pipe or stream.

        Raises:
            ValueError: If called after end().
        """
        if self._ended:
            raise ValueError("cannot write after decoder end")
        # Incremental UTF-8 decode handles multi-byte sequences split across chunks.
        text = self._decoder.decode(chunk, final=False)
        self._consume(text)

    def end(self) -> None:
        """
        Finalize the decoder and flush any remaining pending content.

        Flushes the UTF-8 decoder's internal state (final=True), then emits
        any remaining pending content as a final line (even with no trailing
        newline). After this call, further write() calls will raise ValueError.
        Idempotent: calling end() twice is safe.
        """
        if self._ended:
            return
        # Flush the incremental decoder's own buffer.
        text = self._decoder.decode(b"", final=True)
        self._consume(text)
        self._skip_leading_linefeed = False
        # Emit any remaining pending content as a final line.
        if self._pending or self._omitted_pending_chars > 0:
            self._flush_line()
        self._ended = True

    def _consume(self, text: str) -> None:
        """Process decoded text, searching for line terminators and flushing lines."""
        if not text:
            return

        index = 0
        # Handle the edge case where \\r was at the end of the previous chunk
        # and \\n is at the start of this one (don't count as two line breaks).
        if self._skip_leading_linefeed:
            self._skip_leading_linefeed = False
            if text.startswith("\n"):
                index = 1

        start = index
        while index < len(text):
            character = text[index]
            if character not in ("\r", "\n"):
                index += 1
                continue

            # Found a line terminator; flush everything before it.
            self._append_pending(text[start:index])

            # Special case: \\r at the very end of this chunk.
            # Skip leading \\n in the NEXT chunk (if present).
            if character == "\r" and index == len(text) - 1:
                self._flush_line()
                self._skip_leading_linefeed = True
                return

            self._flush_line()

            # Handle \\r\\n as a single line break, not two.
            if character == "\r" and index + 1 < len(text) and text[index + 1] == "\n":
                index += 1  # Skip the \\n

            index += 1
            start = index

        # Append any remaining text after the last line terminator (or all of it
        # if no terminator was found).
        self._append_pending(text[start:])

    def _append_pending(self, text: str) -> None:
        """Append text to the pending buffer, respecting max_pending_chars."""
        if not text:
            return

        # If the incoming text alone is >= max_pending_chars, we need to truncate
        # and track the omitted count. Keep only the last max_pending_chars.
        if len(text) >= self._max_pending_chars:
            self._omitted_pending_chars += len(self._pending) + len(text) - self._max_pending_chars
            self._pending = text[-self._max_pending_chars :]
            return

        # Append to pending; if total exceeds max, drop from the front.
        self._pending += text
        overflow = len(self._pending) - self._max_pending_chars
        if overflow > 0:
            self._pending = self._pending[overflow:]
            self._omitted_pending_chars += overflow

    def _flush_line(self) -> None:
        """Emit the pending content as one complete line, then reset."""
        if self._omitted_pending_chars > 0:
            line = (
                f"[GreenBoost] {self._omitted_pending_chars} leading characters omitted "
                f"from oversized output line: {self._pending}"
            )
        else:
            line = self._pending
        self._pending = ""
        self._omitted_pending_chars = 0
        self._on_line(line)
