"""Spatially clustered one-second masks for SessionFM pretraining."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F
from queue import Empty, Full, Queue
from threading import Event, Thread
import time


def sessionfm_mask_geometry(coordinates_m: np.ndarray, radius_m: float | None = None) -> dict:
    coords = np.asarray(coordinates_m, dtype=np.float32)
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords[0]
    if coords.ndim != 2 or coords.shape[-1] != 3 or len(coords) < 2:
        raise ValueError("coordinates_m must have shape (channels >= 2, 3)")
    delta = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))
    distances[np.diag_indices_from(distances)] = np.inf
    nearest = distances.min(axis=1)
    median_nearest = float(np.median(nearest))
    radius = 1.5 * median_nearest if radius_m is None else float(radius_m)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("mask radius must be positive and finite")
    key = (coords.shape, coords.tobytes(), radius)
    cached = _NEIGHBOR_CACHE.get(key)
    if cached is None:
        cached = distances <= radius
        np.fill_diagonal(cached, True)
        _NEIGHBOR_CACHE[key] = cached
    return {"radius_m": radius, "median_nearest_neighbor_m": median_nearest,
            "mean_neighbor_count": float((cached.sum(axis=1) - 1).mean()), "neighbors": cached}


def _sessionfm_neighbors(coordinates_m: np.ndarray, radius_m: float) -> np.ndarray:
    return sessionfm_mask_geometry(coordinates_m, radius_m)["neighbors"]


_NEIGHBOR_CACHE: dict[tuple, np.ndarray] = {}


class SessionFMMaskGenerator:
    """CPU NumPy mask generator with a cached adjacency matrix per channel group."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def generate(self, coordinates_m: np.ndarray, batch_size: int, *,
                 n_blocks: int = 8, mask_fraction: float = 0.55,
                 channel_drop_sample_fraction: float = 0.10,
                 radius_m: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        coords = np.asarray(coordinates_m, dtype=np.float32)
        if coords.ndim == 3 and coords.shape[0] == 1:
            coords = coords[0]
        if coords.ndim != 2 or coords.shape[-1] != 3:
            raise ValueError("coordinates_m must have shape (channels, 3)")
        if n_blocks != 8:
            raise ValueError("SessionFM uses exactly 8 half-second blocks")
        if not 0 <= channel_drop_sample_fraction <= 1 or not 0 < mask_fraction < 1:
            raise ValueError("mask fractions must be between zero and one")
        channels = coords.shape[0]
        segments = n_blocks // 2
        neighbors = sessionfm_mask_geometry(coords, radius_m)["neighbors"]
        masks = np.zeros((batch_size, channels, n_blocks), dtype=np.bool_)
        dropped = np.zeros(batch_size, dtype=np.bool_)
        n_dropped = round(batch_size * channel_drop_sample_fraction)
        if n_dropped:
            dropped[self.rng.permutation(batch_size)[:n_dropped]] = True
        target = round(mask_fraction * channels * segments)
        max_attempts = max(4, channels * segments * 2)

        drop_indices = np.flatnonzero(dropped)
        if len(drop_indices):
            fractions = self.rng.uniform(0.25, 0.75, size=len(drop_indices))
            n_channels = np.maximum(1, np.rint(channels * fractions).astype(np.int32))
            priorities = self.rng.random((len(drop_indices), channels))
            rank = np.argsort(np.argsort(priorities, axis=1), axis=1)
            dropped_channels = rank < n_channels[:, None]
            masks[drop_indices] = np.repeat(dropped_channels[:, :, None], n_blocks, axis=2)

        active_indices = np.flatnonzero(~dropped)
        clustered = np.zeros((len(active_indices), channels, segments), dtype=np.bool_)
        for _ in range(max_attempts):
            counts = clustered.sum(axis=(1, 2))
            active = counts < target
            if not active.any():
                break
            rows = np.flatnonzero(active)
            centers = self.rng.integers(channels, size=len(rows))
            segment_ids = self.rng.integers(segments, size=len(rows))
            candidates = neighbors[centers] & ~clustered[rows, :, segment_ids]
            budgets = target - counts[rows]
            overflow = candidates.sum(axis=1) > budgets
            if overflow.any():
                priorities = self.rng.random(candidates.shape)
                order = np.argsort(priorities, axis=1)
                ranks = np.argsort(order, axis=1)
                candidates[overflow] &= ranks[overflow] < budgets[overflow, None]
            clustered[rows[:, None], np.arange(channels)[None, :], segment_ids[:, None]] |= candidates

        counts = clustered.sum(axis=(1, 2))
        flat = clustered.reshape(len(active_indices), -1)
        for row in np.flatnonzero(counts < target):
            remaining = np.flatnonzero(~flat[row])
            flat[row, self.rng.permutation(remaining)[:target - counts[row]]] = True
        masks[active_indices] = np.repeat(clustered, 2, axis=-1)
        return masks, dropped


class SessionFMMaskPrefetcher:
    """Bounded per-channel-group background queues for CPU mask generation."""

    def __init__(self, coordinates_by_group: dict, *, batch_size: int, seed: int,
                 queue_depth: int = 2, state: dict | None = None):
        self.batch_size = batch_size
        self.coordinates = {key: np.asarray(value, dtype=np.float32)
                            for key, value in coordinates_by_group.items()}
        self.queues = {key: Queue(maxsize=queue_depth) for key in self.coordinates}
        self.generators = {key: SessionFMMaskGenerator(seed + index * 100003)
                           for index, key in enumerate(self.coordinates)}
        self.geometry = {key: sessionfm_mask_geometry(value)
                         for key, value in self.coordinates.items()}
        self.generation_seconds = {key: 0.0 for key in self.coordinates}
        if state is not None:
            for key, generator_state in state["generators"].items():
                self.generators[key].rng.bit_generator.state = generator_state
            self.generation_seconds.update(state.get("generation_seconds", {}))
            for key, batches in state["queues"].items():
                for batch in batches:
                    self.queues[key].put_nowait(batch)
        self.stop_event = Event()
        self._start_thread()

    def _start_thread(self):
        self.stop_event.clear()
        self.thread = Thread(target=self._run, name="sessionfm-mask-prefetch", daemon=True)
        self.thread.start()

    def _run(self):
        keys = list(self.coordinates)
        while not self.stop_event.is_set():
            progressed = False
            for key in keys:
                if self.stop_event.is_set():
                    return
                try:
                    started = time.perf_counter()
                    generated = self.generators[key].generate(self.coordinates[key], self.batch_size)
                    self.generation_seconds[key] += time.perf_counter() - started
                    self.queues[key].put_nowait(generated)
                    progressed = True
                except Full:
                    continue
            if not progressed:
                self.stop_event.wait(0.002)

    def get(self, group):
        return self.queues[group].get()

    def close(self):
        self.stop_event.set()
        self.thread.join()

    def snapshot_and_restart(self) -> dict:
        """Freeze producer state and queued batches at an epoch boundary."""
        self.stop_event.set()
        self.thread.join()
        state = {
            "generators": {key: generator.rng.bit_generator.state
                           for key, generator in self.generators.items()},
            "queues": {key: [] for key in self.queues},
            "generation_seconds": dict(self.generation_seconds),
        }
        for key, queue in self.queues.items():
            while True:
                try:
                    state["queues"][key].append(queue.get_nowait())
                except Empty:
                    break
        for key, batches in state["queues"].items():
            for batch in batches:
                self.queues[key].put_nowait(batch)
        self._start_thread()
        return state

    def close_with_state(self) -> dict:
        """Stop the producer and return a resumable, queue-consistent snapshot."""
        self.stop_event.set()
        self.thread.join()
        return {
            "generators": {key: generator.rng.bit_generator.state
                           for key, generator in self.generators.items()},
            "queues": {key: [] for key in self.queues},
            "generation_seconds": dict(self.generation_seconds),
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def sessionfm_spatiotemporal_mask_reference(
    coordinates_m: torch.Tensor,
    *,
    batch_size: int,
    n_blocks: int = 8,
    mask_fraction: float = 0.55,
    channel_drop_sample_fraction: float = 0.10,
    radius_m: float = 0.03,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (batch, channels, blocks) mask and per-sample channel-drop flags.

    Spatial clusters mask neighboring electrodes for the same one-second
    segment. Channel-drop samples mask every time block for selected channels.
    """
    if coordinates_m.ndim == 3:
        if coordinates_m.shape[0] != 1:
            raise ValueError("coordinates must be (channels, 3) or (1, channels, 3)")
        coordinates_m = coordinates_m[0]
    if coordinates_m.ndim != 2 or coordinates_m.shape[-1] != 3:
        raise ValueError("coordinates_m must have shape (channels, 3)")
    if n_blocks != 8:
        raise ValueError("SessionFM uses exactly 8 half-second blocks")
    if not 0 <= channel_drop_sample_fraction <= 1 or not 0 < mask_fraction < 1:
        raise ValueError("mask fractions must be between zero and one")
    device = coordinates_m.device
    channels = coordinates_m.shape[0]
    segments = n_blocks // 2
    distances = torch.cdist(coordinates_m.float(), coordinates_m.float())
    neighbors = distances <= radius_m
    mask = torch.zeros((batch_size, channels, n_blocks), dtype=torch.bool, device=device)
    dropped = torch.zeros(batch_size, dtype=torch.bool, device=device)
    n_dropped = round(batch_size * channel_drop_sample_fraction)
    if n_dropped:
        dropped[torch.randperm(batch_size, device=device, generator=generator)[:n_dropped]] = True

    for sample in range(batch_size):
        if dropped[sample]:
            n_channels = max(1, round(channels * torch.empty((), device=device).uniform_(0.25, 0.75, generator=generator).item()))
            selected = torch.randperm(channels, device=device, generator=generator)[:n_channels]
            mask[sample, selected] = True
            continue

        target = round(mask_fraction * channels * segments)
        clustered = torch.zeros((channels, segments), dtype=torch.bool, device=device)
        for _ in range(max(4, channels * segments * 2)):
            if int(clustered.sum()) >= target:
                break
            center = int(torch.randint(channels, (), device=device, generator=generator))
            segment = int(torch.randint(segments, (), device=device, generator=generator))
            candidates = torch.nonzero(neighbors[center] & ~clustered[:, segment], as_tuple=False).flatten()
            budget = target - int(clustered.sum())
            if candidates.numel() > budget:
                candidates = candidates[torch.randperm(candidates.numel(), device=device, generator=generator)[:budget]]
            clustered[candidates, segment] = True
        if int(clustered.sum()) < target:
            remaining = torch.nonzero(~clustered.flatten(), as_tuple=False).flatten()
            take = min(target - int(clustered.sum()), remaining.numel())
            if take:
                chosen = remaining[torch.randperm(remaining.numel(), device=device, generator=generator)[:take]]
                clustered.view(-1)[chosen] = True
        mask[sample] = clustered.repeat_interleave(2, dim=-1)
    return mask, dropped


def sessionfm_spatiotemporal_mask(
    coordinates_m: torch.Tensor | np.ndarray,
    *,
    batch_size: int,
    n_blocks: int = 8,
    mask_fraction: float = 0.55,
    channel_drop_sample_fraction: float = 0.10,
    radius_m: float | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility API backed by the vectorized CPU NumPy generator."""
    if isinstance(coordinates_m, torch.Tensor):
        device = coordinates_m.device
        coords = coordinates_m.detach().cpu().numpy()
    else:
        device = torch.device("cpu")
        coords = np.asarray(coordinates_m)
    seed = int(torch.randint(0, 2**31 - 1, (), generator=generator).item()) if generator else int(
        np.random.SeedSequence().generate_state(1)[0]
    )
    masks, dropped = SessionFMMaskGenerator(seed).generate(
        coords, batch_size, n_blocks=n_blocks, mask_fraction=mask_fraction,
        channel_drop_sample_fraction=channel_drop_sample_fraction, radius_m=radius_m,
    )
    return (torch.from_numpy(masks).to(device=device),
            torch.from_numpy(dropped).to(device=device))


def sessionfm_masked_trivial_losses(
    wave_target: torch.Tensor,
    spectrum_target: torch.Tensor,
    global_target: torch.Tensor,
    block_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute same-batch masked losses for fixed trivial predictors.

    The waveform predictor is zero. Spectral predictors use one median per
    feature over the masked positions in the current batch.
    """
    if wave_target.ndim != 4 or wave_target.shape[:3] != block_mask.shape:
        raise ValueError("wave_target and block_mask must have shapes (B,C,8,60) and (B,C,8)")
    expected = (block_mask.shape[0], block_mask.shape[1], 4)
    if spectrum_target.shape[:3] != expected or global_target.shape != spectrum_target.shape:
        raise ValueError("spectral targets must both have shape (B,C,4,F)")
    segment_mask = block_mask.reshape(*block_mask.shape[:2], 4, 2).all(-1)
    if not bool(block_mask.any()) or not bool(segment_mask.any()):
        raise ValueError("trivial masked losses require at least one masked block and segment")
    wave_mask = block_mask[..., None].to(wave_target.dtype)
    wave_denominator = (wave_mask.sum() * wave_target.shape[-1]).clamp_min(1)
    wave_loss = (wave_target.abs() * wave_mask).sum() / wave_denominator
    spectral_mask = segment_mask[..., None]
    spectral_count = (spectral_mask.sum() * spectrum_target.shape[-1]).clamp_min(1)

    def median_loss(target: torch.Tensor) -> torch.Tensor:
        selected = target[segment_mask]
        constant = selected.median(dim=0).values
        error = F.smooth_l1_loss(constant.expand_as(selected), selected, reduction="sum")
        return error / spectral_count

    return {"wave": wave_loss, "spectrum": median_loss(spectrum_target),
            "global": median_loss(global_target)}


def sessionfm_mask_statistics(mask: np.ndarray | torch.Tensor,
                              channel_dropped: np.ndarray | torch.Tensor,
                              coordinates_m: np.ndarray | torch.Tensor) -> dict:
    mask_array = (mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor)
                  else np.asarray(mask, dtype=np.bool_))
    dropped_array = (channel_dropped.detach().cpu().numpy() if isinstance(channel_dropped, torch.Tensor)
                     else np.asarray(channel_dropped, dtype=np.bool_))
    coords = (coordinates_m.detach().cpu().numpy() if isinstance(coordinates_m, torch.Tensor)
              else np.asarray(coordinates_m, dtype=np.float32))
    geometry = sessionfm_mask_geometry(coords)
    segments = mask_array.reshape(len(mask_array), mask_array.shape[1], 4, 2).all(axis=-1)
    neighbors = geometry["neighbors"].copy()
    np.fill_diagonal(neighbors, False)
    focal = segments[:, :, None, :]
    co_masked = segments[:, None, :, :]
    denominator = np.count_nonzero(focal & neighbors[None, :, :, None])
    numerator = np.count_nonzero(focal & co_masked & neighbors[None, :, :, None])
    return {
        "radius_m": geometry["radius_m"],
        "median_nearest_neighbor_m": geometry["median_nearest_neighbor_m"],
        "mean_neighbor_count": geometry["mean_neighbor_count"],
        "mask_fraction": float(mask_array.mean()),
        "channel_drop_sample_fraction": float(dropped_array.mean()),
        "neighbor_comask_rate": float(numerator / max(denominator, 1)),
        "independent_mask_baseline": float(segments.mean()),
    }


__all__ = ["SessionFMMaskGenerator", "SessionFMMaskPrefetcher", "sessionfm_spatiotemporal_mask",
           "sessionfm_spatiotemporal_mask_reference", "sessionfm_masked_trivial_losses",
           "sessionfm_mask_geometry", "sessionfm_mask_statistics"]
