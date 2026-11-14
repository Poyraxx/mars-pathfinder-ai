"""CAHVORE projection for the actual calibrated rover camera, without a guessed FOV.

Numerical projection adapted to NumPy from NASA-AMMOS VICAR cmod_cahvore.c,
cmod_cahvore_3d_to_2d_general (Todd Litwin, California Institute of Technology,
copyright 1998-2022), distributed by NASA-AMMOS under Apache-2.0.
https://github.com/NASA-AMMOS/VICAR/blob/master/vos/p2/sub/cahvor/cmod_cahvore.c
Partial derivatives are omitted. PDS camera coordinates are zero-based pixels.
"""
from __future__ import annotations

import numpy as np


def project(points: np.ndarray, model: dict) -> np.ndarray:
    """Project N×3 points in the model's declared coordinate frame to pixels.

    Non-convergent, behind-camera and outside-model rays yield NaN, never a
    plausible-looking mirrored line. Cropping to the image is the caller's job.
    """
    c, a, h, v, o, r, e = [np.asarray(model[k], dtype=float) for k in 'CAHVORE']
    if model['type'] != 'CAHVORE':
        raise ValueError('Unsupported camera model')
    mtype = int(model['T'])
    linearity = {1: 1., 2: 0., 3: float(model['P'])}.get(mtype)
    if linearity is None:
        raise ValueError('Unsupported CAHVORE projection type')
    pc = np.atleast_2d(points).astype(float) - c
    zeta = pc @ o
    lam3 = pc - zeta[:, None] * o
    lam = np.linalg.norm(lam3, axis=1)
    theta = np.arctan2(lam, zeta)
    converged = np.zeros(len(pc), dtype=bool)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        for _ in range(100):
            sn, cs = np.sin(theta), np.cos(theta)
            pupil = e[0] + e[1]*theta**2 + e[2]*theta**4
            deriv = zeta*cs + lam*sn - (1-cs)*pupil - (theta-sn)*(2*e[1]*theta + 4*e[2]*theta**3)
            delta = np.divide(zeta*sn-lam*cs-(theta-sn)*pupil, deriv,
                              out=np.full_like(theta, np.nan), where=np.abs(deriv)>1e-15)
            converged |= np.abs(delta)<1e-8
            theta = np.where(converged, theta, theta-delta)
            if converged.all():
                break
        chi = (np.tan(linearity*theta)/linearity if linearity>1e-15 else
               np.sin(linearity*theta)/linearity if linearity<-1e-15 else theta)
        zetap = np.divide(lam, chi, out=zeta.copy(), where=np.abs(chi)>1e-15)
        mu = r[0] + r[1]*chi**2 + r[2]*chi**4
        rp = zetap[:, None]*o + (1+mu[:, None])*lam3
        rp[theta<1e-8] = pc[theta<1e-8]
        alpha = rp @ a
        uv = np.column_stack((rp@h/alpha, rp@v/alpha))
    good = converged & (pc@a>0) & (alpha>1e-12) & (theta*np.abs(linearity)<np.pi/2-1e-12)
    uv[~good] = np.nan
    return uv


def site_points_to_pixels(points: np.ndarray, model: dict) -> np.ndarray:
    rotation = np.asarray(model['rover_to_site_rotation'])
    origin = np.asarray(model['rover_origin_in_site_m'])
    return project((points-origin)@rotation, model)
