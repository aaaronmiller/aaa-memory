#!/usr/bin/env python3
"""
Clawdi Memory Capture Bridge — Local HTTP server for Tampermonkey scripts.

Listens on localhost:8787 for POST /capture requests from Tampermonkey
and writes them to ~/knowledge/raw/web/<platform>/*.jsonl

Run as: python3 scripts/capture-bridge.py &
"""

import http.server
import socketserver
import json
import os
import time
from datetime import datetime
from pathlib import Path

OUTPUT_BASE = Path("/home/misscheta/knowledge/raw/web")
PORT = 8787


class CaptureHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quiet — no console spam
        pass

    def do_POST(self):
        if self.path != "/capture":
            self.send_error(404, "Not found")
            return

        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len)

        try:
            data = json.loads(body)
            platform = data.get("platform", "unknown")
            turn = data.get("turn", data)  # accept {turn: {...}} or raw turn

            # Ensure platform directory exists
            platform_dir = OUTPUT_BASE / platform
            platform_dir.mkdir(parents=True, exist_ok=True)

            # Write as JSONL
            ts = int(time.time() * 1000)
            rand = os.urandom(3).hex()
            filename = platform_dir / f"{ts}-{rand}.jsonl"

            with open(filename, "w") as f:
                f.write(json.dumps(turn) + "\n")

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        except Exception as e:
            self.send_error(500, str(e))


def run():
    with socketserver.TCPServer(("", PORT), CaptureHandler) as httpd:
        print(f"[Clawdi] Capture bridge listening on :{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[Clawdi] Capture bridge stopped")


if __name__ == "__main__":
    run()
