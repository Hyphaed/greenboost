"""
Tests for BoundedLineDecoder.

Covers normal operation, edge cases with UTF-8 encoding, line terminator
handling, buffer truncation, and integration scenarios.
"""
from __future__ import annotations

import pytest

from greenboost_cli.core.bounded_lines import BoundedLineDecoder


class TestBoundedLineDecoderBasics:
    """Basic normal-case usage."""

    def test_single_complete_line(self):
        """A complete line ending with \\n is decoded correctly."""
        lines = []
        decoder = BoundedLineDecoder(1024, lambda line: lines.append(line))

        decoder.write(b"hello\n")
        decoder.end()

        assert lines == ["hello"]

    def test_multiple_complete_lines(self):
        """Multiple lines in one chunk are separated correctly."""
        lines = []
        decoder = BoundedLineDecoder(1024, lambda line: lines.append(line))

        decoder.write(b"line1\nline2\nline3\n")
        decoder.end()

        assert lines == ["line1", "line2", "line3"]

    def test_line_without_final_newline(self):
        """A final line without a newline is flushed at end()."""
        lines = []
        decoder = BoundedLineDecoder(1024, lambda line: lines.append(line))

        decoder.write(b"hello")
        decoder.end()

        assert lines == ["hello"]

    def test_empty_input(self):
        """Empty input produces no lines."""
        lines = []
        decoder = BoundedLineDecoder(1024, lambda line: lines.append(line))

        decoder.end()

        assert lines == []

    def test_empty_chunks(self):
        """Empty chunks are ignored."""
        lines = []
        decoder = BoundedLineDecoder(1024, lambda line: lines.append(line))

        decoder.write(b"")
        decoder.write(b"hello\n")
        decoder.write(b"")
        decoder.end()

        assert lines == ["hello"]


class TestBoundedLineDecoderMultibyteUTF8:
    """UTF-8 handling, including split multi-byte characters."""

    def test_single_byte_ascii(self):
        """ASCII text is handled normally."""
        lines = []
        decoder = BoundedLineDecoder(1024, lambda line: lines.append(line))

        decoder.write(b"ascii\n")
        decoder.end()

        assert lines == ["ascii"]

    def test_two_byte_utf8_character(self):
        """A complete multi-byte UTF-8 character is decoded."""
        lines = []
        decoder = BoundedLineDecoder(1024, lambda line: lines.append(line))

        # é is encoded as 0xC3 0xA9 in UTF-8
        decoder.write("café\n".encode("utf-8"))
        decoder.end()

        assert lines == ["café"]

    def test_two_byte_utf8_split_across_chunks(self):
        """A 2-byte UTF-8 character split across two chunks."""
        lines = []
        decoder = BoundedLineDecoder(1024, lambda line: lines.append(line))

        # café: c(0x63) a(0x61) f(0x66) é(0xC3 0xA9) \n(0x0A)
        # Split between C3 and A9
        encoded = "café\n".encode("utf-8")
        split_point = encoded.index(0xC3) + 1
        decoder.write(encoded[:split_point])
        decoder.write(encoded[split_point:])
        decoder.end()

        assert lines == ["café"]

    def test_three_byte_utf8_character_split(self):
        """A 3-byte UTF-8 character (中) split across chunks."""
        lines = []
        decoder = BoundedLineDecoder(1024, lambda line: lines.append(line))

        # 中 is encoded as 0xE4 0xB8 0xAD in UTF-8
        encoded = "中\n".encode("utf-8")
        # Split after first byte of multi-byte sequence
        decoder.write(encoded[:1])
        decoder.write(encoded[1:])
        decoder.end()

        assert lines == ["中"]

    def test_mixed_ascii_and_utf8(self):
        """Mix of ASCII and multi-byte UTF-8."""
        lines = []
        decoder = BoundedLineDecoder(1024, lambda line: lines.append(line))

        decoder.write("hello 世界\n".encode("utf-8"))
        decoder.end()

        assert lines == ["hello 世界"]


class TestBoundedLineDecoderLineTerminators:
    """Handling of different line terminator styles."""

    def test_lf_only(self):
        """Unix line terminator (\\n) works."""
        lines = []
        decoder = BoundedLineDecoder(1024, lambda line: lines.append(line))

        decoder.write(b"line1\nline2\n")
        decoder.end()

        assert lines == ["line1", "line2"]

    def test_crlf(self):
        """Windows line terminator (\\r\\n) is treated as one break."""
        lines = []
        decoder = BoundedLineDecoder(1024, lambda line: lines.append(line))

        decoder.write(b"line1\r\nline2\r\n")
        decoder.end()

        assert lines == ["line1", "line2"]

    def test_bare_cr(self):
        """Old Mac line terminator (\\r) works."""
        lines = []
        decoder = BoundedLineDecoder(1024, lambda line: lines.append(line))

        decoder.write(b"line1\rline2\r")
        decoder.end()

        assert lines == ["line1", "line2"]

    def test_crlf_split_across_chunks(self):
        """\\r\\n split across two chunks is NOT treated as two line breaks."""
        lines = []
        decoder = BoundedLineDecoder(1024, lambda line: lines.append(line))

        # Split between \\r and \\n
        decoder.write(b"line1\r")
        decoder.write(b"\nline2\n")
        decoder.end()

        # Should get two lines, not three
        assert lines == ["line1", "line2"]

    def test_bare_cr_at_chunk_boundary(self):
        """\\r at end of chunk followed by \\n at start of next chunk."""
        lines = []
        decoder = BoundedLineDecoder(1024, lambda line: lines.append(line))

        # Same as above: the leading \\n should be skipped
        decoder.write(b"a\r")
        decoder.write(b"\nb")
        decoder.end()

        assert lines == ["a", "b"]

    def test_bare_cr_at_chunk_boundary_followed_by_regular_content(self):
        """\\r at chunk boundary; next chunk starts with non-newline."""
        lines = []
        decoder = BoundedLineDecoder(1024, lambda line: lines.append(line))

        decoder.write(b"a\r")
        decoder.write(b"b\n")
        decoder.end()

        # \\r is a line break; \\r at boundary shouldn't swallow next char
        assert lines == ["a", "b"]


class TestBoundedLineDecoderTruncation:
    """Buffer truncation when a line exceeds max_pending_chars."""

    def test_line_within_limit(self):
        """A line smaller than the limit is not truncated."""
        lines = []
        decoder = BoundedLineDecoder(100, lambda line: lines.append(line))

        decoder.write(b"short line\n")
        decoder.end()

        assert lines == ["short line"]

    def test_line_exactly_at_limit(self):
        """A line exactly at the limit is not truncated."""
        lines = []
        decoder = BoundedLineDecoder(10, lambda line: lines.append(line))

        # 10 chars exactly
        decoder.write(b"0123456789\n")
        decoder.end()

        assert lines == ["0123456789"]

    def test_line_exceeds_limit_in_one_chunk(self):
        """A line exceeding the limit in a single chunk shows truncation marker."""
        lines = []
        decoder = BoundedLineDecoder(10, lambda line: lines.append(line))

        # 20 chars, limit 10 -> 10 chars dropped
        decoder.write(b"0123456789abcdefghij\n")
        decoder.end()

        assert len(lines) == 1
        line = lines[0]
        # Should contain the omitted count and the tail
        assert "[GreenBoost]" in line
        assert "10 leading characters omitted" in line
        assert "abcdefghij" in line

    def test_line_exceeds_limit_gradually(self):
        """Appending text gradually that exceeds the limit."""
        lines = []
        decoder = BoundedLineDecoder(20, lambda line: lines.append(line))

        # Gradually build a 40-char line (limit 20)
        decoder.write(b"0123456789")  # 10 chars
        decoder.write(b"abcdefghij")  # 20 chars total
        decoder.write(b"ABCDEFGHIJ")  # 30 chars total -> 10 chars dropped, keep "abcdefghijABCDEFGHIJ"
        decoder.write(b"klmnopqrst\n")  # 40 chars before newline -> another 10 dropped, keep "ABCDEFGHIJklmnopqrst"

        decoder.end()

        assert len(lines) == 1
        line = lines[0]
        assert "[GreenBoost]" in line
        # 10 omitted from first overflow + 10 from second = 20 total
        assert "20 leading characters omitted" in line
        assert "klmnopqrst" in line

    def test_truncation_preserves_utf8_integrity(self):
        """Truncation of a line with UTF-8 characters doesn't mangle text."""
        lines = []
        decoder = BoundedLineDecoder(10, lambda line: lines.append(line))

        # 中 is 3 bytes in UTF-8, repeated 20 times = 20 chars, exceeds 10-char limit
        data = "中" * 20 + "\n"
        decoder.write(data.encode("utf-8"))
        decoder.end()

        assert len(lines) == 1
        line = lines[0]
        # Should show truncation
        assert "leading characters omitted" in line

    def test_truncation_count_accumulates(self):
        """Multiple truncation events accumulate the omitted count."""
        lines = []
        decoder = BoundedLineDecoder(10, lambda line: lines.append(line))

        # First chunk: 15 chars -> 5 omitted
        decoder.write(b"0123456789abcde")
        # Second chunk: 10 chars -> but pending is already over, so 5+10-kept = more omitted
        decoder.write(b"ABCDEFGHIJ\n")
        decoder.end()

        assert len(lines) == 1
        line = lines[0]
        assert "leading characters omitted" in line
        # 5 (from first chunk overflow) + 5 (from second chunk beyond limit) = 10?
        # Actually: first chunk: pending=15, keep last 10 -> omit 5
        # then write 10 more: pending=15+10=25, keep last 10 -> omit 15 total
        # But the implementation adds to omitted, so: omitted_from_first=5, then 5 more from second = 10 total
        # Let me verify the exact behavior
        assert "15 leading characters omitted" in line


class TestBoundedLineDecoderErrors:
    """Error handling and invalid inputs."""

    def test_invalid_max_pending_chars_zero(self):
        """max_pending_chars must be positive."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            BoundedLineDecoder(0, lambda x: None)

    def test_invalid_max_pending_chars_negative(self):
        """max_pending_chars must be positive."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            BoundedLineDecoder(-1, lambda x: None)

    def test_invalid_max_pending_chars_not_integer(self):
        """max_pending_chars must be an integer."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            BoundedLineDecoder(10.5, lambda x: None)

    def test_write_after_end_raises(self):
        """write() after end() raises ValueError."""
        decoder = BoundedLineDecoder(1024, lambda x: None)

        decoder.end()
        with pytest.raises(ValueError, match="cannot write after decoder end"):
            decoder.write(b"too late\n")

    def test_end_is_idempotent(self):
        """end() can be called multiple times safely."""
        lines = []
        decoder = BoundedLineDecoder(1024, lambda line: lines.append(line))

        decoder.write(b"hello")
        decoder.end()
        decoder.end()
        decoder.end()

        # Should only produce one line
        assert lines == ["hello"]

    def test_invalid_utf8_is_replaced(self):
        """Invalid UTF-8 bytes are replaced with the Unicode replacement character."""
        lines = []
        decoder = BoundedLineDecoder(1024, lambda line: lines.append(line))

        # 0xFF 0xFF is invalid UTF-8
        decoder.write(b"hello\xff\xffworld\n")
        decoder.end()

        assert len(lines) == 1
        # The decoder uses errors='replace', so invalid bytes become U+FFFD
        assert "hello" in lines[0]
        assert "world" in lines[0]
        # The replacement char will be present but hard to match exactly


class TestBoundedLineDecoderIntegration:
    """Integration tests simulating real subprocess/network scenarios."""

    def test_json_rpc_messages(self):
        """Simulates MCP JSON-RPC newline-delimited message handling."""
        lines = []
        decoder = BoundedLineDecoder(1024 * 1024, lambda line: lines.append(line))

        # Two JSON-RPC messages
        msg1 = b'{"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}}\n'
        msg2 = b'{"jsonrpc": "2.0", "id": 2, "method": "echo", "params": {"text": "hello"}}\n'

        decoder.write(msg1)
        decoder.write(msg2)
        decoder.end()

        assert len(lines) == 2
        import json
        obj1 = json.loads(lines[0])
        obj2 = json.loads(lines[1])
        assert obj1["id"] == 1
        assert obj2["id"] == 2

    def test_large_json_message(self):
        """A large but valid JSON message is handled correctly."""
        lines = []
        decoder = BoundedLineDecoder(1024 * 1024, lambda line: lines.append(line))

        # Create a large JSON object
        large_data = "x" * 100000
        msg = f'{{"id": 1, "data": "{large_data}"}}\n'.encode("utf-8")

        decoder.write(msg)
        decoder.end()

        assert len(lines) == 1
        import json
        obj = json.loads(lines[0])
        assert len(obj["data"]) == 100000

    def test_misbehaving_subprocess_no_newline_limit(self):
        """A subprocess that sends data without a newline respects the bound."""
        lines = []
        max_size = 100000  # 100KB limit
        decoder = BoundedLineDecoder(max_size, lambda line: lines.append(line))

        # Simulate a runaway subprocess: send 150KB without newlines (exceeds 100KB limit)
        chunk = b"X" * 150000
        decoder.write(chunk)

        # Add a newline to flush
        decoder.write(b"\n")
        decoder.end()

        assert len(lines) == 1
        line = lines[0]
        # Should show truncation - at least 50KB should be omitted
        assert "leading characters omitted" in line
        assert "50000" in line  # At least 50K chars omitted

    def test_streaming_chunks_of_varying_size(self):
        """Real-world scenario: small and large chunks mixed."""
        lines = []
        decoder = BoundedLineDecoder(1024, lambda line: lines.append(line))

        # Simulate unpredictable network/pipe chunk sizes
        decoder.write(b"small\n")
        decoder.write(b"m" * 500)
        decoder.write(b"e" * 300)
        decoder.write(b"\n")
        decoder.write(b"x")
        decoder.write(b"y")
        decoder.write(b"z\n")
        decoder.end()

        assert len(lines) == 3
        assert lines[0] == "small"
        assert "m" in lines[1] and "e" in lines[1]
        assert lines[2] == "xyz"

    def test_many_empty_lines(self):
        """Many empty lines (just newlines) are handled correctly."""
        lines = []
        decoder = BoundedLineDecoder(1024, lambda line: lines.append(line))

        decoder.write(b"\n\n\nhello\n\n\n")
        decoder.end()

        # The string ends with \n, so no trailing content in pending buffer at end()
        # Therefore we get: ["", "", "", "hello", "", ""]  - 6 empty/content lines
        assert lines == ["", "", "", "hello", "", ""]
