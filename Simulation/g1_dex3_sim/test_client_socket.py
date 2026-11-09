# TL;DR: CPU-Test: PolicyClient baut seinen REQ-Socket nach Timeout neu auf (Review S1, kein EFSM).
"""python -m pytest Simulation/g1_dex3_sim/test_client_socket.py

Ein lokaler REP-Server beantwortet die ERSTE Anfrage absichtlich nicht. Vorher stand der
REQ-Socket danach im falschen Zustand, jedes weitere send() warf EFSM.
"""

from __future__ import annotations

import threading

import pytest


zmq = pytest.importorskip("zmq")
pytest.importorskip("msgpack")

from client import MsgSerializer, PolicyClient  # noqa: E402


def _server(url_holder: list, stop: threading.Event, swallow_first: int):
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.ROUTER)   # ROUTER: darf eine Anfrage unbeantwortet lassen
    port = sock.bind_to_random_port("tcp://127.0.0.1")
    url_holder.append(f"tcp://127.0.0.1:{port}")
    sock.setsockopt(zmq.RCVTIMEO, 100)
    seen = 0
    while not stop.is_set():
        try:
            ident, empty, payload = sock.recv_multipart()
        except zmq.Again:
            continue
        seen += 1
        if seen <= swallow_first:
            continue
        req = MsgSerializer.from_bytes(payload)
        sock.send_multipart([ident, empty, MsgSerializer.to_bytes({"pong": req["endpoint"]})])
    sock.close(0)


@pytest.fixture
def server():
    def start(swallow_first: int):
        url, stop = [], threading.Event()
        t = threading.Thread(target=_server, args=(url, stop, swallow_first), daemon=True)
        t.start()
        while not url:
            pass
        started.append((stop, t))
        return url[0]
    started: list = []
    yield start
    for stop, t in started:
        stop.set()
        t.join(timeout=2)


def test_timeout_then_next_call_works(server):
    client = PolicyClient(server_url=server(swallow_first=1), timeout_ms=300)
    with pytest.raises(zmq.Again):
        client._send_recv({"endpoint": "get_action"})
    # Vorher: zmq.error.ZMQError EFSM. Jetzt: neuer Socket, Antwort kommt.
    assert client._send_recv({"endpoint": "ping"}) == {"pong": "ping"}
    client.close()


def test_ping_survives_a_swallowed_first_request(server):
    client = PolicyClient(server_url=server(swallow_first=1), timeout_ms=300)
    assert client.ping(retries=3, delay=0.0)
    client.close()
