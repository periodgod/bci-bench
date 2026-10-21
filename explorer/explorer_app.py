"""WYU-BCI Bench Explorer: desktop window that browses the bench's modules and code.

    python explorer_app.py              # native window (pywebview / WebView2)
    python explorer_app.py --browser    # local server + default browser
    python explorer_app.py --serve 8765 # local server only (for screenshots / testing)

At start-up it re-scans the bench source on disk, so the view always matches the
code. If the source tree cannot be found, the snapshot bundled at build time is used.
Set WYU_BENCH_ROOT to point at another checkout.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
UI = HERE / "ui"
SNAPSHOT = HERE / "catalog_snapshot.json"
DEFAULT_ROOT = Path(r"D:\脑电大模型\wyu-bci-bench")
ANIMATION_ROOT = Path(r"E:\世界机器人大赛\projects\脑电大模型\animations")
sys.path.insert(0, str(HERE))
from build_catalog import build_catalog  # noqa: E402


def find_root() -> Path | None:
    candidates = []
    if os.environ.get("WYU_BENCH_ROOT"):
        candidates.append(Path(os.environ["WYU_BENCH_ROOT"]))
    start = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    candidates += [start, *start.parents, start / "wyu-bci-bench", *(p / "wyu-bci-bench" for p in start.parents), DEFAULT_ROOT]
    for c in candidates:
        if (c / "src" / "wyu_bci_bench" / "__init__.py").is_file():
            return c
    return None


class Api:
    def __init__(self):
        self.root = find_root()
        self._catalog = None

    def _load(self, fresh: bool) -> dict:
        if self.root is not None:
            cat = build_catalog(self.root)
            cat["snapshot"] = False
            return cat
        if SNAPSHOT.exists():
            cat = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
            cat["snapshot"] = True
            return cat
        raise FileNotFoundError("找不到 bench 源码，也没有内置快照；请设置 WYU_BENCH_ROOT")

    def get_catalog(self):
        if self._catalog is None:
            self._catalog = self._load(False)
        return self._catalog

    def rescan(self):
        self.root = find_root() or self.root
        self._catalog = self._load(True)
        return self._catalog

    def animation_path(self, animation_id: str) -> Path:
        animation = next((item for item in self.get_catalog().get("animations", [])
                          if item["id"] == animation_id), None)
        if animation is None:
            raise FileNotFoundError(f"未知动画：{animation_id}")
        target = Path(animation["video"]).resolve()
        media_root = (ANIMATION_ROOT / "out" / "methods").resolve()
        if media_root not in target.parents or target.suffix.lower() != ".mp4":
            raise PermissionError(f"动画路径无效：{animation_id}")
        if not target.is_file():
            raise FileNotFoundError(f"动画视频尚未渲染：{animation_id}")
        return target

    def open_file(self, rel: str):
        if self.root is None:
            raise RuntimeError("离线快照模式下无法打开源码文件")
        target = (self.root / rel).resolve()
        if self.root.resolve() not in target.parents or not target.is_file():
            raise PermissionError(f"只允许打开 bench 目录内的文件：{rel}")
        os.startfile(str(target))  # type: ignore[attr-defined]  # Windows only
        return True


def make_server(api: Api, port: int = 0) -> ThreadingHTTPServer:
    types = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = self.path.split("?", 1)[0]
            if route.startswith("/animation/"):
                animation_id = route.removeprefix("/animation/")
                try:
                    target = api.animation_path(animation_id)
                    size = target.stat().st_size
                    byte_range = self.headers.get("Range")
                    start, end, status = 0, size - 1, 200
                    if byte_range:
                        match = re.fullmatch(r"bytes=(\d*)-(\d*)", byte_range.strip())
                        if not match or (not match.group(1) and not match.group(2)):
                            self.send_error(416)
                            return
                        if match.group(1):
                            start = int(match.group(1))
                            end = int(match.group(2)) if match.group(2) else size - 1
                        else:
                            suffix_size = int(match.group(2))
                            start = max(0, size - suffix_size)
                        if start >= size or end < start:
                            self.send_error(416)
                            return
                        end = min(end, size - 1)
                        status = 206
                    length = end - start + 1
                    self.send_response(status)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Content-Length", str(length))
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Cache-Control", "no-store")
                    if status == 206:
                        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.end_headers()
                    with target.open("rb") as media:
                        media.seek(start)
                        remaining = length
                        while remaining:
                            chunk = media.read(min(1024 * 1024, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (FileNotFoundError, PermissionError) as exc:
                    self._send(404, str(exc).encode("utf-8"), "text/plain; charset=utf-8")
                return
            name = route.lstrip("/") or "index.html"
            target = (UI / name).resolve()
            if UI.resolve() not in target.parents or not target.is_file():
                return self._send(404, b"not found", "text/plain")
            self._send(200, target.read_bytes(), types.get(target.suffix, "application/octet-stream"))

        def do_POST(self):
            name = self.path.removeprefix("/api/")
            if name not in ("get_catalog", "rescan", "open_file"):
                return self._send(404, b"unknown api", "text/plain")
            try:
                args = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"[]")
                result = getattr(api, name)(*args)
                self._send(200, json.dumps(result, ensure_ascii=False).encode("utf-8"))
            except Exception as exc:
                self._send(500, str(exc).encode("utf-8"), "text/plain; charset=utf-8")

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)  # loopback only: no remote access


def run_browser(api: Api, port: int = 0, open_browser: bool = True) -> None:
    server = make_server(api, port)
    url = f"http://127.0.0.1:{server.server_address[1]}/index.html"
    print(url, flush=True)
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    server.serve_forever()


def run_window(api: Api) -> None:
    import webview

    server = make_server(api)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/index.html"
    try:
        webview.create_window("WYU-BCI Bench Explorer", url, js_api=api,
                              width=1280, height=820, min_size=(900, 600), maximized=True)
        webview.start(gui="edgechromium")
    finally:
        server.shutdown()
        server.server_close()


def main() -> None:
    args = sys.argv[1:]
    api = Api()
    if "--dump" in args:
        out = Path(args[args.index("--dump") + 1])
        out.write_text(json.dumps(api.get_catalog(), ensure_ascii=False), encoding="utf-8")
        print(json.dumps(api.get_catalog()["stats"], ensure_ascii=False))
        return
    if "--serve" in args:
        i = args.index("--serve")
        port = int(args[i + 1]) if i + 1 < len(args) and args[i + 1].isdigit() else 0
        return run_browser(api, port, open_browser=False)
    if "--browser" in args:
        return run_browser(api)
    try:
        run_window(api)
    except Exception as exc:  # WebView2 missing or pywebview unavailable
        print(f"原生窗口不可用（{exc}），改用浏览器打开", flush=True)
        run_browser(api)


if __name__ == "__main__":
    main()
