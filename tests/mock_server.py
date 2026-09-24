"""A tiny stand-in for the APIMaster image/video endpoints, for tests.

Standard library only, so CI needs nothing installed.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

# A 2x2 PNG, so tests never need to generate one.
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAF0lEQVQIW2P8z8Dwn4EIwDiqkL4K"
    "6asQAF9VA/1ZkP1sAAAAAElFTkSuQmCC"
)

STATE = {"image_polls": 0, "video_polls": 0, "mode": "ok"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep test output readable
        pass

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth_ok(self):
        header = self.headers.get("Authorization", "")
        if not header.startswith("Bearer ") or len(header) < 12:
            self._json(401, {"error": {"message": "invalid api key"}})
            return False
        return True

    def do_GET(self):
        if not self._auth_ok():
            return
        if self.path.startswith("/v1/models"):
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": "gpt-image-2", "object": "model"},
                        {"id": "sora-2", "object": "model"},
                        {"id": "gpt-5.5", "object": "model"},
                    ],
                },
            )
            return
        if self.path.startswith("/v1/tasks/"):
            STATE["image_polls"] += 1
            if STATE["image_polls"] < 2:
                self._json(200, {"data": {"status": "pending"}})
            else:
                self._json(
                    200,
                    {
                        "data": {
                            "status": "completed",
                            "result": {"images": [{"url": ["http://127.0.0.1:%d/img/a.png" % self.server.server_port]}]},
                        }
                    },
                )
            return
        if self.path.startswith("/v1/videos/") and self.path.endswith("/content"):
            body = b"\x00\x00\x00\x18ftypmp42fake-mp4-bytes"
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/v1/videos/"):
            STATE["video_polls"] += 1
            status = "in_progress" if STATE["video_polls"] < 2 else "completed"
            payload = {"id": "task_mock", "status": status}
            if status == "completed":
                payload["url"] = "http://127.0.0.1:%d/v1/videos/task_mock/content" % self.server.server_port
            self._json(200, payload)
            return
        if self.path.startswith("/img/"):
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(TINY_PNG)))
            self.end_headers()
            self.wfile.write(TINY_PNG)
            return
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        # Read the body before anything else. Replying and closing with unread request
        # data in the buffer makes Windows send a TCP RST, so the client sometimes sees
        # ConnectionAbortedError (10053) instead of our 401. Found as a 1-in-6 flake.
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        if not self._auth_ok():
            return
        payload = json.loads(raw or b"{}")

        if self.path.startswith("/v1/images/generations/async"):
            self._json(200, {"code": 200, "data": [{"status": "submitted", "task_id": "task_mock_image"}]})
            return
        if self.path.startswith("/v1/images/generations"):
            if STATE["mode"] == "timeout":
                self._json(408, {"error": {"message": "image generation timed out", "code": "image_generation_timeout"}})
                return
            if not payload.get("prompt"):
                self._json(400, {"error": {"message": "prompt is required"}})
                return
            count = int(payload.get("n", 1))
            self._json(
                200,
                {
                    "created": int(time.time()),
                    "data": [
                        {"url": "http://127.0.0.1:%d/img/%d.png" % (self.server.server_port, i)}
                        for i in range(count)
                    ],
                },
            )
            return
        if self.path.startswith("/v1/videos/generations"):
            self._json(200, {"code": 200, "data": [{"status": "submitted", "task_id": "task_mock"}]})
            return
        self._json(404, {"error": {"message": "not found"}})


def start(port: int = 0, mode: str = "ok"):
    STATE.update({"image_polls": 0, "video_polls": 0, "mode": mode})
    server = HTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


if __name__ == "__main__":
    srv = start(8791)
    print(f"mock on http://127.0.0.1:{srv.server_port}/v1")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        srv.shutdown()
