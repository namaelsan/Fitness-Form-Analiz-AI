"""Adaptive smoothing window for pose angle data.

Instead of a fixed number of frames, the smoother targets a fixed *time span*
(``target_window_ms``) and derives how many frames that corresponds to from an
exponential moving average (EMA) of the observed inter-frame interval.  This
means:

* A fast model (60 FPS → 16 ms/frame) uses a larger window (~9 frames for a
  150 ms target) to stay smooth.
* A slow model (8 FPS → 125 ms/frame) uses a smaller window (~2–3 frames) so
  it does not average across stale data.
* Transitions are gradual thanks to the EMA, so temporary latency spikes do
  not cause the window to thrash.

Usage::

    smoother = AdaptiveSmoother()           # sensible defaults
    smoother.update(interval_ms=16.7)       # call once per frame
    window = smoother.window_size           # read the current frame count
"""

from __future__ import annotations

import math


# ---- tuneable defaults -------------------------------------------------------
#
# target_window_ms  – Real-time span we want the smoothing window to cover.
#                     150 ms is a good balance: short enough to track fast
#                     movements, long enough to suppress landmark jitter.
#
# ema_alpha         – EMA forgetting factor for the interval estimate.
#                     Lower  → slower to adapt, more stable.
#                     Higher → faster to adapt, can thrash on spikes.
#                     0.15 adapts in ~6 frames, matching typical FPS changes.
#
# seed_interval_ms  – Assumed inter-frame interval before any real data
#                     arrives.  33.3 ms = 30 FPS, the most common default.
#
# min_frames        – Hard floor.  Never average fewer than this many frames;
#                     below 3 the smoothing is essentially useless.
#
# max_frames        – Hard ceiling.  Prevents over-smoothing on very slow
#                     models where a large window would blur real movement.
# -----------------------------------------------------------------------------

_DEFAULT_TARGET_MS: float = 150.0
_DEFAULT_EMA_ALPHA: float = 0.15
_DEFAULT_SEED_MS: float = 33.3     # 30 FPS
_DEFAULT_MIN_FRAMES: int = 3
_DEFAULT_MAX_FRAMES: int = 15


class AdaptiveSmoother:
    """Derives a dynamic smoothing window from observed inter-frame timing.

    Parameters
    ----------
    target_window_ms:
        How many milliseconds of history the smoothing window should span.
    ema_alpha:
        EMA forgetting factor applied to each new interval sample.
        Must be in (0, 1].  Lower values react more slowly to FPS changes.
    seed_interval_ms:
        Initial interval estimate used before any real frames have arrived.
    min_frames:
        Minimum window size (inclusive).
    max_frames:
        Maximum window size (inclusive).
    """

    def __init__(
        self,
        target_window_ms: float = _DEFAULT_TARGET_MS,
        ema_alpha: float = _DEFAULT_EMA_ALPHA,
        seed_interval_ms: float = _DEFAULT_SEED_MS,
        min_frames: int = _DEFAULT_MIN_FRAMES,
        max_frames: int = _DEFAULT_MAX_FRAMES,
    ) -> None:
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        if min_frames < 1:
            raise ValueError(f"min_frames must be >= 1, got {min_frames}")
        if max_frames < min_frames:
            raise ValueError("max_frames must be >= min_frames")

        self.target_window_ms = target_window_ms
        self.ema_alpha = ema_alpha
        self.min_frames = min_frames
        self.max_frames = max_frames

        # Seed the EMA so we have a sensible window from frame 1.
        self._ema_interval_ms: float = seed_interval_ms
        # Track whether any real data has been observed yet.
        self._warmed_up: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, interval_ms: float) -> None:
        """Record a new inter-frame interval and update the EMA.

        Parameters
        ----------
        interval_ms:
            Time elapsed since the previous frame, in milliseconds.
            Values <= 0 are ignored (e.g. first frame, duplicate timestamps).
        """
        if interval_ms <= 0.0:
            return

        if not self._warmed_up:
            # On the very first real observation, hard-reset the EMA to the
            # actual value so we do not drag the seed through the estimate.
            self._ema_interval_ms = interval_ms
            self._warmed_up = True
        else:
            self._ema_interval_ms = (
                self.ema_alpha * interval_ms
                + (1.0 - self.ema_alpha) * self._ema_interval_ms
            )

    @property
    def estimated_fps(self) -> float:
        """Current FPS estimate derived from the EMA interval."""
        return 1000.0 / self._ema_interval_ms

    @property
    def estimated_interval_ms(self) -> float:
        """Current inter-frame interval estimate in milliseconds."""
        return self._ema_interval_ms

    @property
    def window_size(self) -> int:
        """Number of frames the smoothing window should span right now.

        Computed as ``round(target_window_ms / ema_interval_ms)``, then
        clamped to ``[min_frames, max_frames]``.
        """
        raw = self.target_window_ms / self._ema_interval_ms
        return max(self.min_frames, min(self.max_frames, math.ceil(raw)))

    def reset(self) -> None:
        """Reset the EMA back to the seed state (e.g. on model/source switch)."""
        self._ema_interval_ms = _DEFAULT_SEED_MS
        self._warmed_up = False

    def __repr__(self) -> str:
        return (
            f"AdaptiveSmoother("
            f"window={self.window_size}, "
            f"fps≈{self.estimated_fps:.1f}, "
            f"target={self.target_window_ms:.0f}ms)"
        )
