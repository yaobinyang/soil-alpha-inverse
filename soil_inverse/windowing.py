"""Sliding-window index generation for the soil thermal inverse problem.

A *window* is a contiguous block of resampled time steps over which a single
scalar ``alpha`` is estimated.  Windows of length ``window_hours`` are emitted at
``stride_hours`` intervals starting from the first timestamp.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd


def _as_timestamps(times: Sequence) -> pd.DatetimeIndex:
    """Coerce an arbitrary time container into a sorted ``DatetimeIndex``."""
    idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(times)))
    return idx


def make_window_indices(
    times: Sequence,
    window_hours: float,
    stride_hours: float,
    *,
    min_points: int = 3,
) -> List[np.ndarray]:
    """Return a list of integer index arrays, one per sliding window.

    Parameters
    ----------
    times
        Monotonically increasing timestamps (any pandas-parseable form).
    window_hours
        Length of each window in hours.  A window spans ``[t_start, t_start +
        window_hours]`` inclusive.
    stride_hours
        Spacing between consecutive window starts, in hours.
    min_points
        Windows containing fewer than this many samples are dropped; this guards
        the finite-difference solver against degenerate (1-2 point) windows.

    Notes
    -----
    The window start advances by exactly ``stride_hours`` from the first
    timestamp, independent of the (possibly irregular) sample spacing, so the
    number of windows is ``floor((span - window) / stride) + 1`` whenever the
    data are gap-free.
    """
    idx = _as_timestamps(times)
    if len(idx) == 0:
        return []

    values = idx.values  # datetime64[ns], vectorised comparisons
    t0 = values[0]
    t_end = values[-1]

    window = np.timedelta64(int(round(window_hours * 3600.0)), "s")
    stride = np.timedelta64(int(round(stride_hours * 3600.0)), "s")
    if stride <= np.timedelta64(0, "s"):
        raise ValueError("stride_hours must be positive")

    windows: List[np.ndarray] = []
    start = t0
    # A tiny tolerance lets the final window land exactly on the last sample.
    tol = np.timedelta64(1, "s")
    while start + window <= t_end + tol:
        stop = start + window
        members = np.flatnonzero((values >= start) & (values <= stop))
        if members.size >= min_points:
            windows.append(members)
        start = start + stride
    return windows


def select_windows_by_time(windows: Sequence[np.ndarray], times: Sequence, start=None, end=None) -> List[np.ndarray]:
    """Return windows whose center timestamp lies inside ``[start, end]``.

    This helper is retained for notebooks written before the YB module update.
    ``start`` and ``end`` may be ``None`` or any pandas-parseable timestamp.
    """
    idx = _as_timestamps(times)
    start_ts = pd.Timestamp.min if start is None else pd.Timestamp(start)
    end_ts = pd.Timestamp.max if end is None else pd.Timestamp(end)
    selected: List[np.ndarray] = []
    for window in windows:
        if len(window) == 0:
            continue
        center = idx[int(window[len(window) // 2])]
        if start_ts <= center <= end_ts:
            selected.append(np.asarray(window, dtype=int))
    return selected
