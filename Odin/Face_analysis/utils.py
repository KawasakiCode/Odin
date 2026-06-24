import numpy as np
import math

def acute_angle_between(P1, P2, Q1, Q2):
    """Unsigned angle (deg) between line P1->P2 and line Q1->Q2, folded to [0, 90].

    Uses abs(dot) so the result is independent of how each segment is oriented.
    This makes left/right mirror-image segments directly comparable, which a
    raw signed-angle difference does not.
    """
    v1 = np.asarray(P2, dtype=float) - np.asarray(P1, dtype=float)
    v2 = np.asarray(Q2, dtype=float) - np.asarray(Q1, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_angle = abs(np.dot(v1, v2)) / (n1 * n2)
    return math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))

def descent_below_horizontal(P1, P2):
    """Angle (deg) the segment P1-P2 makes with the horizontal, folded to [0, 90].

    0 = perfectly horizontal, 90 = vertical. Orientation-independent (uses the
    magnitude of the run and rise), so the left and right jaw segments — which
    point in mirrored directions — both return the same positive slope and can
    be averaged meaningfully.
    """
    dx = abs(P2[0] - P1[0])
    dy = abs(P2[1] - P1[1])
    return math.degrees(math.atan2(dy, dx))
    
