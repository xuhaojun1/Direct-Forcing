"""Numeric chunk-boundary diagnostics; frame numbers are zero based."""

import math


def boundary_statistics(deltas, first_frames):
    """deltas[t] measures frame t -> t+1; first_frames lists new-chunk frames."""
    positions = {frame-1 for frame in first_frames}
    if not positions or min(positions) < 0 or max(positions) >= len(deltas):
        raise ValueError('Chunk boundary outside the video')
    if not all(math.isfinite(value) and value >= 0 for value in deltas):
        raise ValueError('Expected finite nonnegative transition magnitudes')
    boundary = [value for i, value in enumerate(deltas) if i in positions]
    interior = [value for i, value in enumerate(deltas) if i not in positions]
    if not interior:
        raise ValueError('Need interior transitions for comparison')
    b, w = sum(boundary)/len(boundary), sum(interior)/len(interior)
    return {'boundary_mean': b, 'interior_mean': w, 'ratio': b/max(w, 1e-12),
            'first_frames': list(first_frames), 'transition_magnitudes': list(deltas)}
