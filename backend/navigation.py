"""Deterministic terrain analysis and bounded, optimal eight-connected A*."""
import heapq
import math
import time
import numpy as np


def terrain_analysis(elevation: np.ndarray, cell_m: float) -> dict:
    valid = np.isfinite(elevation)
    dy, dx = np.gradient(elevation.astype(np.float64), cell_m)
    slope = np.degrees(np.arctan(np.hypot(dx, dy)))
    valid &= np.isfinite(slope)
    roughness = np.hypot(np.gradient(dx, cell_m)[1], np.gradient(dy, cell_m)[0]) * cell_m
    return {"slope": slope, "roughness": roughness, "valid": valid}


def cost_grid(analysis: dict, mode: str, max_slope: float, blocked: np.ndarray | None = None) -> np.ndarray:
    valid = analysis["valid"].copy()
    cost = np.ones(valid.shape, dtype=np.float64)
    if mode != "distance":
        valid &= analysis["slope"] <= max_slope
        cost += 3 * (analysis["slope"] / max_slope) ** 2 + np.minimum(analysis["roughness"] * 2, 2)
    if blocked is not None:
        valid &= ~blocked
    cost[~valid] = np.inf
    return cost


def astar(cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int], cell_m: float = 1,
          heuristic: bool = True, timeout_s: float = 15) -> tuple[list[tuple[int, int]], float]:
    rows, cols = cost.shape
    for p in (start, goal):
        if not (0 <= p[0] < rows and 0 <= p[1] < cols) or not np.isfinite(cost[p]):
            raise ValueError("endpoint_blocked_or_outside")
    began = time.perf_counter()
    flat = cost.ravel().tolist()
    finite = np.isfinite(cost).ravel().tolist()
    gr, gc = goal
    hypot, heappush, heappop = math.hypot, heapq.heappush, heapq.heappop
    def h(p):
        return hypot(p[0]-gr, p[1]-gc) * cell_m if heuristic else 0
    scores = {start: 0.0}
    parents = {}
    heap = [(h(start), 0.0, start)]
    directions = [(dr, dc, hypot(dr, dc) * cell_m) for dr, dc in
                  [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]]
    while heap:
        _, g, p = heappop(heap)
        if g > scores[p]:
            continue
        if p == goal:
            path = [p]
            while p in parents:
                p = parents[p]
                path.append(p)
            return list(reversed(path)), g
        if time.perf_counter() - began > timeout_s:
            raise TimeoutError("planning_timeout")
        pr, pc = p
        here = flat[pr*cols+pc]
        for dr, dc, step in directions:
            r, c = pr+dr, pc+dc
            if not (0 <= r < rows and 0 <= c < cols):
                continue
            qi = r*cols+c
            if not finite[qi]:
                continue
            if dr and dc and (not finite[(pr+dr)*cols+pc] or not finite[pr*cols+pc+dc]):
                continue
            q = (r, c)
            ng = g + step * (here + flat[qi]) / 2
            if ng < scores.get(q, math.inf):
                scores[q] = ng
                parents[q] = p
                heappush(heap, (ng+h(q), ng, q))
    raise ValueError("no_path")


def plan_legs(cost: np.ndarray, points: list[tuple[int,int]], cell_m: float) -> tuple[list, float]:
    path, total = [], 0.0
    for start, goal in zip(points, points[1:]):
        leg, value = astar(cost, start, goal, cell_m)
        path.extend(leg if not path else leg[1:])
        total += value
    return path, total


def metrics(path: list, elevation: np.ndarray, slope: np.ndarray, cell_m: float, speed: float) -> dict:
    idx = tuple(np.array(path).T)
    heights = elevation[idx]
    lengths = np.hypot(np.diff(np.array(path)[:,0]), np.diff(np.array(path)[:,1])) * cell_m
    distance = float(lengths.sum())
    samples = np.unique(np.linspace(0,len(path)-1,min(100,len(path))).astype(int))
    cumulative = np.r_[0, np.cumsum(lengths)]
    return {"distance_m": round(distance,2), "ascent_m": round(float(np.maximum(np.diff(heights),0).sum()),2),
            "max_slope_deg": round(float(slope[idx].max()),2), "duration_s": round(distance/speed),
            "profile": [{"distance_m":round(float(cumulative[i]),2), "elevation_m":round(float(heights[i]),2)} for i in samples]}


def inflate_obstacles(shape: tuple, points_rc: np.ndarray, radius_cells: float) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    radius = math.ceil(radius_cells)
    for row, col in np.unique(np.rint(points_rc).astype(int), axis=0):
        r0,r1 = max(0,row-radius), min(shape[0],row+radius+1)
        c0,c1 = max(0,col-radius), min(shape[1],col+radius+1)
        if r1 <= r0 or c1 <= c0:
            continue
        yy,xx = np.ogrid[r0:r1,c0:c1]
        mask[r0:r1,c0:c1] |= (yy-row)**2+(xx-col)**2 <= radius_cells**2
    return mask
