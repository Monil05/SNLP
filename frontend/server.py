"""Serve CareerMatch and proxy /api/* requests to the local career_nlp API."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import json
import os

FRONTEND_DIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 5500
BACKEND_URL = "http://127.0.0.1:8000"


class FrontendHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def do_GET(self):
        if urlsplit(self.path).path.startswith("/api/"):
            self._proxy_api()
        else:
            super().do_GET()

    def do_POST(self):
        if urlsplit(self.path).path.startswith("/api/"):
            self._proxy_api()
        else:
            self.send_error(405, "POST is only supported for /api/ requests")

    def _proxy_api(self):
        parsed = urlsplit(self.path)
        api_path = parsed.path[len("/api"):]
        if not api_path.startswith("/v1/"):
            self._send_json(404, {"detail": "Unknown API path."})
            return

        target_url = f"{BACKEND_URL}{api_path}"
        if parsed.query:
            target_url += f"?{parsed.query}"

        body = None
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            body = self.rfile.read(length)

        headers = {"Accept": self.headers.get("Accept", "application/json")}
        content_type = self.headers.get("Content-Type")
        if content_type:
            headers["Content-Type"] = content_type
        api_key = self.headers.get("X-API-Key")
        if api_key:
            headers["X-API-Key"] = api_key

        request = Request(target_url, data=body, headers=headers, method=self.command)
        try:
            with urlopen(request, timeout=180) as response:
                payload = response.read()
                status = response.status
                response_type = response.headers.get("Content-Type", "application/json")
        except HTTPError as error:
            payload = error.read()
            status = error.code
            response_type = error.headers.get("Content-Type", "application/json")
        except (URLError, TimeoutError, OSError) as error:
            reason = getattr(error, "reason", error)
            self._send_json(502, {
                "detail": (
                    f"Could not connect to the backend at {BACKEND_URL}. "
                    "Start the backend in another terminal, then try again. "
                    f"({reason})"
                )
            })
            return

        self.send_response(status)
        self.send_header("Content-Type", response_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, status, value):
        payload = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    os.chdir(FRONTEND_DIR)
    server = ThreadingHTTPServer((HOST, PORT), FrontendHandler)
    print(f"CareerMatch frontend: http://{HOST}:{PORT}")
    print(f"Proxying /api/* requests to {BACKEND_URL}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nFrontend server stopped.")
    finally:
        server.server_close()
