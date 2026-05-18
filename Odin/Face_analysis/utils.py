import numpy as np
import math

def angle_at_vertex(A, B, C):
    """Angle in degrees at point B formed by A-B-C"""
    BA = A - B
    BC = C - B
    cos_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

def line_angle_degrees(P1, P2):
        """Angle of line P1->P2 relative to horizontal in image coords."""
        dx = P2[0] - P1[0]
        dy = P2[1] - P1[1]
        return math.degrees(math.atan2(-dy, dx))  # negate dy for inverted y