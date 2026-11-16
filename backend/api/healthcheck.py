"""CLI helper to check or wait for the VitalAI FastAPI backend server."""

from __future__ import annotations

import argparse
import socket
import sys
import time
import urllib.request


def is_port_open(host: str = "127.0.0.1", port: int = 8000) -> bool:
    """Return True if a TCP service is accepting connections on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def wait_for_health(url: str = "http://127.0.0.1:8000/health", timeout_s: float = 15.0) -> bool:
    """Poll the FastAPI /health endpoint until HTTP 200 or timeout expires."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check or wait for VitalAI FastAPI backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--wait", type=float, default=0.0, help="Seconds to wait for /health.")
    args = parser.parse_args()

    if args.wait > 0:
        url = f"http://{args.host}:{args.port}/health"
        return 0 if wait_for_health(url=url, timeout_s=args.wait) else 1
    return 0 if is_port_open(host=args.host, port=args.port) else 1


if __name__ == "__main__":
    sys.exit(main())
