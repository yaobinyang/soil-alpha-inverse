"""Loading and resampling of the raw soil sensor CSV.

The raw file (e.g. ``Lupit_20210729_0812.csv``) has an irregular ~10 min cadence
with columns ``time, M1..M4, T1..T4, D``.  We resample onto a regular grid so the
finite-difference solver can march at a constant ``dt``.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd


def parse_time(value: str) -> datetime:
    """Parse timestamp strings used by the field CSV files."""
    formats = [
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unsupported datetime format: {value!r}")


def load_temperature_csv(path: Union[str, Path], temp_columns: Sequence[str]):
    """Load a CSV with a time column and selected temperature columns."""
    times = []
    rows = []
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        time_column = next(
            (column for column in (reader.fieldnames or []) if column.lower() == "time"),
            None,
        )
        if time_column is None:
            raise ValueError("CSV must contain a time column")
        for row in reader:
            times.append(parse_time(row[time_column]))
            rows.append([float(row[col]) if row[col] != "" else np.nan for col in temp_columns])
    return np.array(times, dtype=object), np.array(rows, dtype=float)


def load_and_resample_soil_csv(
    csv_path: Union[str, Path],
    temp_columns: Sequence[str],
    water_content_columns: Sequence[str],
    *,
    resample_interval: str = "10min",
    time_column: str = "time",
    aggregation: str = "mean",
    interpolate: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load a soil CSV and resample temperature / water-content channels.

    The raw logger cadence (~11 min) does not divide the resampling interval, so
    some bins are empty.  With ``interpolate=True`` (default) those gaps are
    filled by time-linear interpolation, yielding a gap-free regular series so
    that sliding windows are not silently dropped for containing a NaN.

    Returns
    -------
    times
        ``object`` array of :class:`datetime.datetime` (regular cadence).  Plain
        ``datetime`` objects are used so that timestamp differences expose
        ``.total_seconds()`` as the analysis notebook expects.
    obs_temperatures
        ``float`` array of shape ``(n_times, n_temp_cols)``.
    water_content
        ``float`` array of shape ``(n_times, n_water_cols)``.
    resampled
        The resampled :class:`pandas.DataFrame` (temperature + water columns),
        indexed by the regular time grid.
    """
    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path)
    if time_column not in frame.columns:
        raise KeyError(f"time column {time_column!r} not found in {csv_path}")

    frame[time_column] = pd.to_datetime(frame[time_column])
    frame = frame.set_index(time_column).sort_index()

    temp_columns = list(temp_columns)
    water_content_columns = list(water_content_columns)
    wanted = temp_columns + water_content_columns
    missing = [c for c in wanted if c not in frame.columns]
    if missing:
        raise KeyError(f"columns {missing} not found in {csv_path}")

    numeric = frame[wanted].apply(pd.to_numeric, errors="coerce")
    resampler = numeric.resample(resample_interval)
    resampled = getattr(resampler, aggregation)()
    if interpolate:
        # Fill bins with no raw sample; ``limit_direction="both"`` also covers a
        # leading/trailing empty bin so the series is fully gap-free.
        resampled = resampled.interpolate(method="time", limit_direction="both")

    times = np.array(resampled.index.to_pydatetime(), dtype=object)
    obs_temperatures = resampled[temp_columns].to_numpy(dtype=float)
    water_content = resampled[water_content_columns].to_numpy(dtype=float)
    return times, obs_temperatures, water_content, resampled


def median_time_step_seconds(times: Sequence) -> float:
    """Median spacing between consecutive timestamps, in seconds."""
    idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(times)))
    if len(idx) < 2:
        raise ValueError("need at least two timestamps to infer a time step")
    deltas = np.diff(idx.values).astype("timedelta64[s]").astype(float)
    return float(np.median(deltas))
