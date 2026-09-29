"""TensorBoard launcher patched for WSL2 mirrored networking.

Anomaly (proven experimentally 2026-09-27): on this WSL2 mirrored-mode
stack, a TCP connect to an UNBOUND 127.0.0.1 port is a black hole -- no
RST; it blocks until socket timeout / kernel SYN retries (~2min+).
Unbound LAN-IP ports refuse instantly; bound loopback ports connect
instantly; socket timeouts ARE honored (EAGAIN, rc=11).

TensorBoard's startup pre-check is_port_in_use() (a nested closure in
program.py __init__, unreachable by module-level patching) calls
connect_ex(("localhost", port)) with NO timeout -> hangs for minutes
before werkzeug ever binds. We wrap socket.socket.connect_ex with a 1s
timeout so the pre-check degrades to "port free" (nonzero rc) after 1s.
Semantics preserved for every other caller.

--load_fast=false: the Rust data-server also dies silently on this stack.
"""
import socket
import sys

_real_connect_ex = socket.socket.connect_ex


def _connect_ex_with_timeout(self, address):
    old = self.gettimeout()
    try:
        self.settimeout(1.0)
        return _real_connect_ex(self, address)
    finally:
        self.settimeout(old)


socket.socket.connect_ex = _connect_ex_with_timeout

sys.argv = [
    "tensorboard",
    "--logdir", "exp/timbre_blend_stage1",
    # Loopback only: never expose experiment audio on the LAN.
    "--host", "127.0.0.1",
    "--port", "6006",
    "--reload_interval", "15",
    "--load_fast=false",
    "--window_title", "DDSP-SVC-Stage1",
]
from tensorboard.main import run_main

run_main()
