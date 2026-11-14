import math
import numpy as np
import pytest
from backend.navigation import astar, terrain_analysis, cost_grid, inflate_obstacles


def test_astar_matches_dijkstra_on_weighted_obstacle_grid():
    rng=np.random.default_rng(13)
    for _ in range(5):
        grid=1+rng.random((18,23))*3
        grid[rng.random(grid.shape)<.1]=np.inf
        grid[0,0]=grid[-1,-1]=1
        try:
            path,cost=astar(grid,(0,0),(17,22))
        except ValueError:
            with pytest.raises(ValueError):astar(grid,(0,0),(17,22),heuristic=False)
        else:
            _,reference=astar(grid,(0,0),(17,22),heuristic=False)
            assert cost==pytest.approx(reference)
            assert path[0]==(0,0) and path[-1]==(17,22)


def test_corner_cutting_is_blocked():
    grid=np.array([[1,np.inf],[np.inf,1.]])
    with pytest.raises(ValueError,match="no_path"):astar(grid,(0,0),(1,1))


def test_slope_uses_metres_and_unknown_is_not_zero():
    elevation=np.tile(np.arange(10,dtype=float)*2,(10,1))
    a=terrain_analysis(elevation,2)
    assert np.allclose(a["slope"],45)
    elevation[5,5]=np.nan
    a=terrain_analysis(elevation,2)
    assert not a["valid"][5,5]
    assert not np.isfinite(cost_grid(a,"terrain",20)).any()


def test_new_obstacle_replans_around_real_changed_cells():
    grid=np.ones((20,20))
    before,before_cost=astar(grid,(10,1),(10,18))
    blocked=inflate_obstacles(grid.shape,np.array([[10,10]]),2)
    updated=grid.copy();updated[blocked]=np.inf
    after,after_cost=astar(updated,(10,1),(10,18))
    assert any(blocked[p] for p in before)
    assert not any(blocked[p] for p in after)
    assert after_cost>before_cost


def test_invalid_endpoint_is_rejected():
    with pytest.raises(ValueError,match="endpoint"):astar(np.ones((3,3)),(-1,0),(2,2))
