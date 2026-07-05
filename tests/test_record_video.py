"""Tests for _latest_frame in record_video."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from record_video import _latest_frame


def test_latest_frame_copies_buffer():
    """Copy frame so later buffer edits do not change it."""
    buf = np.zeros((1, 1, 8, 8, 3), dtype=np.uint8)
    frame0 = _latest_frame(buf, 0)
    buf[0, 0, :, :, :] = 255
    assert frame0.sum() == 0


def test_latest_frame_tracks_in_place_updates():
    """Each call reads the buffer at that moment."""
    buf = np.zeros((1, 1, 4, 4, 3), dtype=np.uint8)
    frames = [_latest_frame(buf, 0)]
    for val in (64, 128, 192):
        buf[0, 0, :, :, :] = val
        frames.append(_latest_frame(buf, 0))
    diffs = [
        np.mean(np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16)))
        for i in range(1, len(frames))
    ]
    assert all(d > 0 for d in diffs)
