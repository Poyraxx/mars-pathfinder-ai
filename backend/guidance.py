"""Rover-centred navigation HUD and independent recorded-pose change analysis.

The HUD is a steering cue in normalized screen coordinates, not a camera/ground
registration. The calibrated camera projection remains a separate output.
"""
from __future__ import annotations

import math
import numpy as np

ANCHOR = [.5, .94]
MOVE_THRESHOLD_M = .25
TURN_THRESHOLD_DEG = 5.


def signed_angle(degrees: float) -> float:
    return (degrees + 180.) % 360. - 180.


def pose_change(previous_xy, previous_yaw, xy, yaw) -> dict:
    """Compare actual acquisition poses; never infer motion from the planned route."""
    base = {'source': 'archived PLACES pose change', 'position_threshold_m': MOVE_THRESHOLD_M,
            'heading_threshold_deg': TURN_THRESHOLD_DEG}
    if previous_xy is None:
        return {**base, 'kind': 'initial', 'label': 'First recorded pose',
                'distance_m': None, 'heading_change_deg': None, 'travel_bearing_deg': None}
    delta = np.asarray(xy, dtype=float) - np.asarray(previous_xy, dtype=float)
    distance = float(np.linalg.norm(delta))
    turn = signed_angle(float(yaw) - float(previous_yaw))
    moving, rotating = distance >= MOVE_THRESHOLD_M, abs(turn) >= TURN_THRESHOLD_DEG
    bearing = math.degrees(math.atan2(delta[0], delta[1])) % 360 if moving else None
    if rotating:
        if abs(turn) >= 165:
            kind, label = 'reoriented', 'Reoriented'
        else:
            side = 'right' if turn > 0 else 'left'
            kind = ('turn_' if not moving else '') + side
            label = f"Turned {side}" if moving else f"Rotated {side} in place"
    elif moving:
        relative = signed_angle(bearing - float(previous_yaw))
        if abs(relative) > 120:
            kind, label = 'reverse', 'Moved backward'
        elif abs(relative) >= 60:
            kind = 'right' if relative > 0 else 'left'
            label = f'Moved toward the {kind}'
        else:
            kind, label = 'forward', 'Moved forward'
    else:
        kind, label = 'stationary', 'Position unchanged'
    return {**base, 'kind': kind, 'label': label, 'distance_m': round(distance, 2),
            'heading_change_deg': round(turn, 1),
            'travel_bearing_deg': round(bearing, 1) if bearing is not None else None}


def _bezier(a, b, c, d):
    t = np.linspace(0, 1, 33)[:, None]
    return (1-t)**3*a + 3*(1-t)**2*t*b + 3*(1-t)*t**2*c + t**3*d


def _smooth(points):
    points = np.asarray(points, dtype=float)
    for _ in range(2):
        corners = np.stack((.75*points[:-1]+.25*points[1:], .25*points[:-1]+.75*points[1:]), axis=1)
        points = np.vstack((points[0], corners.reshape(-1, 2), points[-1]))
    return points


def navigation_guide(xy, yaw_deg: float, route_xy, remaining_m: float) -> dict:
    """A bounded HUD, rooted at the rover centre, driven by route-relative bearings.

    Front cues follow up to35 m of the real remaining map path. Side and U-turn
    cues use explicit steering glyphs, rather than mirroring invisible ground.
    """
    origin = np.asarray(xy, dtype=float)
    route = np.asarray(route_xy, dtype=float).reshape(-1, 2)
    path = np.vstack((origin, route))
    path = path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-6]]
    cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    length = float(cumulative[-1])
    def point_at(distance):
        return np.array([np.interp(distance, cumulative, path[:, i]) for i in range(2)])
    target = point_at(min(4., length)) - origin
    turn = signed_angle(math.degrees(math.atan2(target[0], target[1])) - yaw_deg) if np.linalg.norm(target) > .1 else 0.
    anchor = np.asarray(ANCHOR)
    side = 1. if turn >= 0 else -1.
    if remaining_m <= 3.:
        kind, instruction = 'arrive', 'Destination nearby'
        points = _bezier(anchor, np.array([.5, .85]), np.array([.5, .73]), np.array([.5, .64]))
    elif abs(turn) >= 120:
        kind, instruction = 'u_turn', 'Make a U-turn'
        points = _bezier(anchor, np.array([.5, .49]), np.array([.5+side*.44, .50]), np.array([.5+side*.32, .87]))
    elif abs(turn) >= 55:
        kind = 'right' if turn > 0 else 'left'
        instruction = f'Turn {kind}'
        points = _bezier(anchor, np.array([.5, .69]), np.array([.5+side*.22, .60]), np.array([.5+side*.40, .59]))
    else:
        kind = 'ahead' if abs(turn) < 15 else ('bear_right' if turn > 0 else 'bear_left')
        instruction = 'Continue ahead' if kind == 'ahead' else 'Bear '+kind.split('_')[1]
        yaw = math.radians(yaw_deg)
        forward = np.array([math.sin(yaw), math.cos(yaw)])
        right = np.array([math.cos(yaw), -math.sin(yaw)])
        points = [anchor, np.array([.5, .88])]
        for distance in sorted(set([min(s, length) for s in (3., 6., 10., 18., 28., 35.)])):
            delta = point_at(distance)-origin
            ahead, lateral = float(delta@forward), float(delta@right)
            if ahead < .2:
                break
            points.append(np.array([.5+.46*math.tanh(lateral/(ahead+5.)), .94-.64*distance/(distance+7.)]))
        points = _smooth(points)
    return {'anchor': ANCHOR.copy(), 'points': np.round(points, 5).tolist(), 'kind': kind,
            'instruction': instruction, 'turn_deg': round(turn, 1), 'lookahead_m': round(min(35., length), 1),
            'is_ground_projection': False, 'source': 'remaining map route and archived rover heading'}
