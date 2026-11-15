import math

import pytest

from backend import data, replay
from backend.guidance import drive_motion


def line(n, step, yaw, start=(0., 0.)):
    """n drive steps of `step` metres along yaw (clockwise from north); negative step = backward."""
    h = math.radians(yaw)
    return [(start[0]+i*step*math.cos(h), start[1]+i*step*math.sin(h), yaw) for i in range(n+1)]


def test_forward_drive_along_heading():
    m = drive_motion(line(10, .5, 90))
    assert m['kind'] == 'forward' and m['forward_m'] == pytest.approx(5) and m['reverse_m'] == 0
    assert m['label'] == 'Drove 5.0 m forward'


def test_backward_drive_against_heading():
    m = drive_motion(line(6, -.5, 90))
    assert m['kind'] == 'reverse' and m['reverse_m'] == pytest.approx(3)
    assert m['label'].startswith('Backed up 3.0 m')


def test_turn_in_place_right_is_positive_yaw():
    m = drive_motion([(0, 0, 10), (0, 0, 40), (0, 0, 100)])
    assert m['kind'] == 'turn_right' and m['heading_change_deg'] == pytest.approx(90)
    assert m['label'] == 'Turned right 90° in place'


def test_turn_left_across_north_wraps():
    m = drive_motion([(0, 0, 10), (0, 0, -20), (0, 0, -50)])
    assert m['kind'] == 'turn_left' and m['heading_change_deg'] == pytest.approx(-60)


def test_sequence_backup_turn_then_forward():
    seq = line(2, -.5, 0)
    n, e, _ = seq[-1]
    seq += [(n, e, 45), (n, e, 90)]
    seq += line(8, .5, 90, (n, e))[1:]
    m = drive_motion(seq)
    assert [p['kind'] for p in m['phases']] == ['reverse', 'turn', 'forward']
    assert m['kind'] == 'right' and m['reverse_m'] == pytest.approx(1) and m['forward_m'] == pytest.approx(4)
    assert m['detail'] == 'Backed up 1.0 m → turned right 90° in place → drove 4.0 m forward'


def test_stationary():
    assert drive_motion([(1, 1, 30), (1, 1, 30.5)])['kind'] == 'stationary'


def test_real_sol48_drive_is_backward_and_matches_the_tracks():
    """PLACES logs ~40 m of driving against the rover heading on sol 48. Independently, the next
    front Hazcam frame shows fresh wheel tracks ahead of the rover, which only a reverse drive leaves."""
    data.load_regions()
    result = replay.compute('hazcam-extended-traverse')
    step = next(s for s in result['steps'] if s['sol'] == 48 and s['motion'].get('drive_steps', 0) > 50)
    m = step['motion']
    assert m['kind'] == 'reverse' and m['reverse_m'] > 35 and m['forward_m'] < 1
    assert m['phases'][-1]['kind'] == 'turn' and m['phases'][-1]['yaw_deg'] < -80
    assert len(step['driven']) == m['drive_steps']+1


def test_driven_path_connects_consecutive_acquisitions():
    data.load_regions()
    steps = replay.compute('hazcam-extended-traverse')['steps']
    for a, b in zip(steps, steps[1:]):
        if b['driven']:
            assert b['driven'][0] == pytest.approx([a['rover']['lat'], a['rover']['lon']], abs=1e-6)
            assert b['driven'][-1] == pytest.approx([b['rover']['lat'], b['rover']['lon']], abs=1e-6)
