"""Competition package building and static checks.

EEG/EMG Challenge 2026 Track 2 (Codabench): the ZIP root must contain
``submission.py`` plus its weights/config; the model loads files from
``meta['submission_dir']`` and is scored per subject x session stream with
``reset_state()`` called before every session. OmniEEG fixes only the data
reader and scorer interfaces, announced per stage.
"""

from __future__ import annotations

import ast
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Iterable

FORBIDDEN_IN_SUBMISSION = ("os.remove", "shutil.rmtree")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_zip(folder: str | Path, out_zip: str | Path, include: Iterable[str] | None = None) -> dict[str, object]:
    """Zip ``folder``'s files at the archive root (no parent directory) and return a manifest."""
    folder, out_zip = Path(folder), Path(out_zip)
    files = sorted(p for p in folder.iterdir() if p.is_file()) if include is None else [folder / n for n in include]
    if not any(p.name == "submission.py" for p in files):
        raise FileNotFoundError(f"{folder} has no submission.py")
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, arcname=p.name)
    manifest = {"zip": str(out_zip), "bytes": out_zip.stat().st_size, "sha256": _sha256(out_zip.read_bytes()),
                "members": {p.name: _sha256(p.read_bytes()) for p in files}}
    return manifest


def check_zip(path: str | Path, required: Iterable[str] = ("submission.py",)) -> dict[str, object]:
    """Static checks: root layout, exactly one top-level Python entry point, parseable source,
    a ``reset_state`` hook for stream scoring, and no writes into the submission directory."""
    path = Path(path)
    report: dict[str, object] = {"zip": str(path), "ok": True, "problems": []}
    problems: list[str] = report["problems"]  # type: ignore[assignment]
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        report["members"] = names
        for r in required:
            if r not in names:
                problems.append(f"missing {r} at archive root")
        nested = [n for n in names if "/" in n.rstrip("/")]
        if nested:
            problems.append(f"files are nested in folders: {nested[:5]}")
        py = [n for n in names if n.endswith(".py") and "/" not in n]
        if len(py) != 1:
            problems.append(f"expected exactly one root .py file, found {py}")
        if "submission.py" in names:
            src = zf.read("submission.py").decode("utf-8")
            try:
                tree = ast.parse(src)
            except SyntaxError as exc:
                problems.append(f"submission.py does not parse: {exc}")
                tree = None
            if tree is not None:
                funcs = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
                report["functions"] = sorted(funcs)
                if "reset_state" not in funcs:
                    problems.append("no reset_state(): stream scoring resets model state per session")
            if "submission_dir" not in src:
                problems.append("weights are not loaded via meta['submission_dir']")
            for token in FORBIDDEN_IN_SUBMISSION:
                if token in src:
                    problems.append(f"submission.py uses {token}")
    report["ok"] = not problems
    return report


def write_manifest(manifest: dict[str, object], path: str | Path) -> None:
    Path(path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
