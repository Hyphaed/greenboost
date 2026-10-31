"""Esc/Ctrl-C closes the model request at once instead of waiting for a first token."""
from __future__ import annotations

import socket
import threading
import time

import pytest

pytest.importorskip("openai")

from greenboost_cli.inference import adapters as A


def _silent_server():
    """Accepts a connection, reads the request, never answers (a model still reading its prompt)."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    conns = []

    def run():
        c, _ = srv.accept()
        conns.append(c)
        c.recv(65536)
        time.sleep(30)
    threading.Thread(target=run, daemon=True).start()
    return srv, srv.getsockname()[1], conns


def test_abort_unblocks_a_request_that_is_still_waiting_for_the_first_token():
    srv, port, conns = _silent_server()
    outcome = {}

    def worker():
        t0 = time.time()
        try:
            list(A._stream_openai_api("k", f"http://127.0.0.1:{port}/v1", "m", "sys",
                                      [{"role": "user", "content": "hi"}], None, {}))
            outcome["result"] = "finished"
        except BaseException as e:
            outcome["result"] = type(e).__name__
        outcome["secs"] = time.time() - t0
        A.untrack_client()

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    time.sleep(0.8)                                  # the request is out and unanswered
    assert th.is_alive()
    t0 = time.time()
    assert A.abort_active_requests() == 1
    th.join(timeout=5)
    srv.close()
    assert not th.is_alive(), "the worker is still blocked after the abort"
    assert time.time() - t0 < 3
    assert outcome["result"] != "finished"


def test_abort_with_nothing_in_flight_is_harmless():
    assert A.abort_active_requests() == 0


def test_a_cancel_pressed_before_the_request_stops_it_from_being_sent():
    """The first turn spends seconds before the request (embedder load); a cancel in that
    window must stop the turn, not be discovered after the request is already waiting."""
    import threading as _t
    ev = _t.Event()
    ev.set()
    sent = []
    import socket as _s
    srv = _s.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
    port = srv.getsockname()[1]
    _t.Thread(target=lambda: sent.append(srv.accept()), daemon=True).start()
    with pytest.raises(A.TurnCancelled):
        list(A._stream_openai_api("k", f"http://127.0.0.1:{port}/v1", "m", "sys",
                                  [{"role": "user", "content": "hi"}], None, {"_cancel_event": ev}))
    time.sleep(0.3)
    srv.close()
    assert sent == []                                  # nothing connected: no request went out
