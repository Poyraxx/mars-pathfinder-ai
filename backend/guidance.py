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


DRIVE_SOURCE = 'PLACES interpolated drive poses (every drive step between the two acquisitions)'


def drive_motion(samples) -> dict:
    """What the rover actually did between two acquisitions, from its dense localized drive poses.

    samples: [(northing_m, easting_m, yaw_deg), ...] from the previous acquisition pose to this one.
    Each drive step is classified with the rover's own heading at that step: forward (displacement
    along the heading), backward (against it) or a turn in place (yaw change without displacement).
    Yaw is clockwise from north, so a positive change is a right turn.
    """
    base = {'source': DRIVE_SOURCE, 'position_threshold_m': MOVE_THRESHOLD_M, 'heading_threshold_deg': TURN_THRESHOLD_DEG}
    pts = np.asarray([(s[0], s[1]) for s in samples], float)
    yaws = [float(s[2]) for s in samples]
    phases = []
    for i in range(1, len(pts)):
        d = pts[i]-pts[i-1]; dist = float(np.linalg.norm(d)); dyaw = signed_angle(yaws[i]-yaws[i-1])
        mean = math.radians(yaws[i-1]+dyaw/2)
        along = float(d@np.array([math.cos(mean), math.sin(mean)]))
        kind = 'turn' if dist < .05 else ('forward' if along >= 0 else 'reverse')
        if kind == 'turn' and abs(dyaw) < .2:
            continue
        if phases and phases[-1]['kind'] == kind:
            phases[-1]['distance_m'] += dist; phases[-1]['yaw_deg'] += dyaw
        else:
            phases.append({'kind': kind, 'distance_m': dist, 'yaw_deg': dyaw})
    clean = []
    for p in phases:
        if (p['kind'] == 'turn' and abs(p['yaw_deg']) < 3) or (p['kind'] != 'turn' and p['distance_m'] < .1):
            if clean: clean[-1]['yaw_deg'] += p['yaw_deg']
            continue
        if clean and clean[-1]['kind'] == p['kind']:
            clean[-1]['distance_m'] += p['distance_m']; clean[-1]['yaw_deg'] += p['yaw_deg']
        else:
            clean.append(dict(p))
    forward = sum(p['distance_m'] for p in clean if p['kind'] == 'forward')
    reverse = sum(p['distance_m'] for p in clean if p['kind'] == 'reverse')
    net_yaw = sum(signed_angle(yaws[i]-yaws[i-1]) for i in range(1, len(yaws)))
    chord = float(np.linalg.norm(pts[-1]-pts[0])) if len(pts) > 1 else 0.
    moved, turned = forward+reverse >= MOVE_THRESHOLD_M, abs(net_yaw) >= TURN_THRESHOLD_DEG
    side = 'right' if net_yaw > 0 else 'left'
    if not moved and not turned:
        kind = 'stationary'
    elif not moved:
        kind = 'turn_'+side
    elif reverse > forward:
        kind = 'reverse'
    elif turned:
        kind = side
    else:
        kind = 'forward'

    def phrase(p):
        y, s = p['yaw_deg'], 'right' if p['yaw_deg'] > 0 else 'left'
        if p['kind'] == 'turn':
            return f"turned {s} {abs(y):.0f}° in place"
        verb = 'drove' if p['kind'] == 'forward' else 'backed up'
        tail = ' forward' if p['kind'] == 'forward' else ''
        curve = f" curving {s} {abs(y):.0f}°" if abs(y) >= 10 else ''
        return f"{verb} {p['distance_m']:.1f} m{tail}{curve}"
    net = f"{'right' if net_yaw > 0 else 'left'} {abs(net_yaw):.0f}°"
    if kind == 'stationary':
        label = 'Position unchanged'
    elif kind.startswith('turn_'):
        label = f'Turned {net} in place'
    elif kind == 'reverse':
        label = f'Backed up {reverse:.1f} m' + (f', net turn {net}' if turned else '')
    else:
        label = f'Drove {forward:.1f} m forward' + (f', net turn {net}' if turned else '') + (f' (incl. {reverse:.1f} m backward)' if reverse >= MOVE_THRESHOLD_M else '')
    steps_text = ' → '.join(phrase(p) for p in clean)
    detail = (steps_text[0].upper()+steps_text[1:]) if steps_text else 'No drive steps between these acquisitions'
    bearing = math.degrees(math.atan2(pts[-1][1]-pts[0][1], pts[-1][0]-pts[0][0])) % 360 if chord >= MOVE_THRESHOLD_M else None
    return {**base, 'kind': kind, 'label': label, 'detail': detail, 'distance_m': round(forward+reverse, 2), 'net_distance_m': round(chord, 2),
            'forward_m': round(forward, 2), 'reverse_m': round(reverse, 2), 'heading_change_deg': round(net_yaw, 1),
            'travel_bearing_deg': round(bearing, 1) if bearing is not None else None, 'drive_steps': len(samples)-1,
            'phases': [{'kind': p['kind'], 'distance_m': round(p['distance_m'], 2), 'yaw_deg': round(p['yaw_deg'], 1)} for p in clean]}
