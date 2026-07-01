"""Dependency-light SVG plots for offline EEG analysis."""

from __future__ import annotations

from collections import Counter
from html import escape
from pathlib import Path
from typing import Sequence

import numpy as np

from .signal import BANDS, bandpower_from_psd, compute_psd


COLORS = ("#2f5f9f", "#b24b43", "#4c8a4b", "#8a5aa8", "#b67d2c", "#2c8c8c", "#555555", "#c64f7f")

CHANNEL_POSITIONS = {
    "FC3": (-0.38, 0.42),
    "FCZ": (0.0, 0.45),
    "FC4": (0.38, 0.42),
    "C3": (-0.48, 0.0),
    "C1": (-0.2, 0.0),
    "CZ": (0.0, 0.0),
    "C2": (0.2, 0.0),
    "C4": (0.48, 0.0),
    "CP3": (-0.38, -0.42),
    "CPZ": (0.0, -0.45),
    "CP4": (0.38, -0.42),
    "FP1": (-0.28, 0.82),
    "FP2": (0.28, 0.82),
    "F3": (-0.45, 0.6),
    "F4": (0.45, 0.6),
    "P3": (-0.45, -0.6),
    "P4": (0.45, -0.6),
    "O1": (-0.28, -0.82),
    "O2": (0.28, -0.82),
}


def _svg_path(path: str | Path) -> Path:
    out = Path(path)
    if out.suffix.lower() != ".svg":
        out = out.with_suffix(".svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _write(path: Path, body: str, width: int, height: int) -> Path:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#ffffff"/>\n'
        f"{body}\n"
        "</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")
    return path


def _text(x: float, y: float, value: object, size: int = 12, anchor: str = "middle", color: str = "#222") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Microsoft YaHei, sans-serif" '
        f'font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(str(value))}</text>'
    )


def _scale(values: np.ndarray, low: float, high: float, invert: bool = True) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full(arr.shape, (low + high) / 2.0)
    scaled = low + (arr - lo) * (high - low) / (hi - lo)
    if invert:
        scaled = high - (scaled - low)
    return scaled


def _color(t: float, diverging: bool = False) -> str:
    t = max(0.0, min(1.0, float(t)))
    if diverging:
        stops = ((49, 103, 180), (247, 247, 247), (178, 24, 43))
    else:
        stops = ((52, 94, 141), (37, 145, 132), (245, 187, 64))
    left, right = (stops[0], stops[1]) if t <= 0.5 else (stops[1], stops[2])
    u = t * 2.0 if t <= 0.5 else (t - 0.5) * 2.0
    rgb = [round(left[i] + (right[i] - left[i]) * u) for i in range(3)]
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _color_values(values: np.ndarray, diverging: bool = False) -> list[str]:
    arr = np.asarray(values, dtype=float)
    if diverging:
        bound = max(float(np.max(np.abs(arr))), 1e-12)
        t = (arr + bound) / (2.0 * bound)
    else:
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        t = (arr - lo) / max(hi - lo, 1e-12)
    return [_color(float(item), diverging=diverging) for item in t.reshape(-1)]


def plot_label_distribution(labels: Sequence[object], out_path: str | Path) -> Path:
    path = _svg_path(out_path)
    counts = Counter(str(label) for label in labels)
    names = list(counts.keys())
    values = np.asarray([counts[name] for name in names], dtype=float)
    width = max(640, 90 * max(1, len(names)) + 120)
    height = 420
    left, right, top, bottom = 70, 30, 50, 90
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_value = max(float(np.max(values)), 1.0)
    gap = 12
    bar_w = max(18, (chart_w - gap * (len(names) + 1)) / max(1, len(names)))
    body = [_text(width / 2, 28, "Label distribution", size=18)]
    body.append(f'<line x1="{left}" y1="{top+chart_h}" x2="{width-right}" y2="{top+chart_h}" stroke="#333"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}" stroke="#333"/>')
    for idx, (name, value) in enumerate(zip(names, values)):
        x = left + gap + idx * (bar_w + gap)
        h = chart_h * float(value) / max_value
        y = top + chart_h - h
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{COLORS[idx % len(COLORS)]}"/>')
        body.append(_text(x + bar_w / 2, top + chart_h + 22, name, size=11))
        body.append(_text(x + bar_w / 2, y - 6, int(value), size=11))
    body.append(_text(20, top + chart_h / 2, "Trials", size=12, anchor="start"))
    return _write(path, "\n".join(body), width, height)


def plot_psd(
    eeg: np.ndarray,
    fs: int,
    out_path: str | Path,
    y: np.ndarray | None = None,
    label_names: Sequence[str] | None = None,
    max_hz: float = 45.0,
) -> Path:
    path = _svg_path(out_path)
    freqs, psd = compute_psd(eeg, fs)
    mask = freqs <= float(max_hz)
    freqs = freqs[mask]
    curves: list[tuple[str, np.ndarray]] = []
    if y is None:
        curves.append(("all", np.mean(psd[..., mask], axis=(0, 1))))
    else:
        labels = np.asarray(y, dtype=np.int64)
        for label in sorted(set(int(v) for v in labels)):
            key = label_names[label] if label_names and label < len(label_names) else str(label)
            curves.append((key, np.mean(psd[labels == label][..., mask], axis=(0, 1))))

    width, height = 820, 470
    left, right, top, bottom = 70, 170, 50, 60
    chart_w = width - left - right
    chart_h = height - top - bottom
    all_values = np.concatenate([np.log10(np.maximum(curve, 1e-18)) for _, curve in curves])
    x_pos = left + (freqs - float(freqs[0])) * chart_w / max(float(freqs[-1] - freqs[0]), 1e-12)
    body = [_text(width / 2, 28, "Average spectrum", size=18)]
    body.append(f'<rect x="{left}" y="{top}" width="{chart_w}" height="{chart_h}" fill="#fafafa" stroke="#333"/>')
    for tick in range(0, int(max_hz) + 1, 10):
        x = left + tick * chart_w / max_hz
        body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+chart_h}" stroke="#e0e0e0"/>')
        body.append(_text(x, top + chart_h + 20, tick, size=11))
    for idx, (name, curve) in enumerate(curves):
        log_curve = np.log10(np.maximum(curve, 1e-18))
        y_pos = _scale(log_curve, top + 10, top + chart_h - 10, invert=True)
        points = " ".join(f"{x:.1f},{yy:.1f}" for x, yy in zip(x_pos, y_pos))
        color = COLORS[idx % len(COLORS)]
        body.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_y = top + 18 + idx * 20
        body.append(f'<line x1="{width-right+25}" y1="{legend_y}" x2="{width-right+48}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        body.append(_text(width - right + 55, legend_y + 4, name, size=11, anchor="start"))
    body.append(_text(left + chart_w / 2, height - 18, "Frequency (Hz)", size=12))
    body.append(_text(10, top + chart_h / 2, "log10 PSD", size=12, anchor="start"))
    body.append(_text(width - 38, height - 20, f"range {float(np.min(all_values)):.1f}..{float(np.max(all_values)):.1f}", size=10, anchor="end", color="#666"))
    return _write(path, "\n".join(body), width, height)


def plot_band_energy_heatmap(
    eeg: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
    out_path: str | Path,
) -> Path:
    path = _svg_path(out_path)
    freqs, psd = compute_psd(eeg, fs)
    band_names, powers = bandpower_from_psd(freqs, psd, BANDS)
    energy = np.log10(np.maximum(np.mean(powers, axis=0), 1e-18))
    rows, cols = energy.shape
    cell_w, cell_h = 92, 28
    left, top = 95, 55
    width = left + cols * cell_w + 40
    height = top + rows * cell_h + 80
    colors = _color_values(energy, diverging=False)
    body = [_text(width / 2, 28, "Log band energy by channel", size=18)]
    for col, name in enumerate(band_names):
        body.append(_text(left + col * cell_w + cell_w / 2, top - 12, name, size=10))
    flat_idx = 0
    for row, channel in enumerate(channel_names):
        y = top + row * cell_h
        body.append(_text(left - 10, y + 18, channel, size=10, anchor="end"))
        for col in range(cols):
            x = left + col * cell_w
            body.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{colors[flat_idx]}" stroke="#ffffff"/>')
            flat_idx += 1
    return _write(path, "\n".join(body), width, height)


def _channel_positions(channel_names: Sequence[str]) -> list[tuple[float, float]]:
    positions: list[tuple[float, float]] = []
    missing: list[int] = []
    for idx, name in enumerate(channel_names):
        key = "".join(ch for ch in str(name).upper() if ch.isalnum())
        if key in CHANNEL_POSITIONS:
            positions.append(CHANNEL_POSITIONS[key])
        else:
            positions.append((0.0, 0.0))
            missing.append(idx)
    if missing:
        for pos, idx in enumerate(missing):
            angle = 2.0 * np.pi * pos / max(1, len(missing))
            positions[idx] = (0.72 * float(np.cos(angle)), 0.72 * float(np.sin(angle)))
    return positions


def plot_topomap_or_bar(
    values: np.ndarray,
    channel_names: Sequence[str],
    out_path: str | Path,
    title: str = "Channel map",
) -> Path:
    path = _svg_path(out_path)
    vals = np.asarray(values, dtype=float).reshape(-1)
    positions = _channel_positions(channel_names)
    colors = _color_values(vals, diverging=True)
    width, height = 520, 500
    cx, cy, radius = width / 2, height / 2 + 10, 180
    body = [_text(width / 2, 32, title, size=18)]
    body.append(f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="#f7f7f7" stroke="#333" stroke-width="2"/>')
    body.append(f'<path d="M {cx-28:.1f} {cy-radius+4:.1f} Q {cx:.1f} {cy-radius-35:.1f} {cx+28:.1f} {cy-radius+4:.1f}" fill="none" stroke="#333" stroke-width="2"/>')
    for (x_norm, y_norm), name, color, value in zip(positions, channel_names, colors, vals):
        x = cx + float(x_norm) * radius
        y = cy - float(y_norm) * radius
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="28" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
        body.append(_text(x, y + 4, name, size=10))
        body.append(_text(x, y + 42, f"{value:.2g}", size=9, color="#555"))
    body.append(_text(width / 2, height - 18, "blue: lower in class 2, red: higher in class 2", size=11, color="#666"))
    return _write(path, "\n".join(body), width, height)


def plot_class_band_topomap(
    eeg: np.ndarray,
    y: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
    out_path: str | Path,
    band: tuple[float, float] = (8.0, 12.0),
) -> Path | None:
    labels = np.asarray(y, dtype=np.int64)
    classes = sorted(set(int(v) for v in labels))
    if len(classes) < 2:
        return None
    freqs, psd = compute_psd(eeg, fs)
    _, powers = bandpower_from_psd(freqs, psd, {"band": band})
    first = np.mean(powers[labels == classes[0], :, 0], axis=0)
    second = np.mean(powers[labels == classes[1], :, 0], axis=0)
    diff = np.log10(np.maximum(second, 1e-18)) - np.log10(np.maximum(first, 1e-18))
    return plot_topomap_or_bar(diff, channel_names, out_path, title=f"{band[0]:g}-{band[1]:g} Hz class difference")
