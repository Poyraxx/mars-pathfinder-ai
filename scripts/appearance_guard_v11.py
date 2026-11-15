"""Research-only conservative guard; always intersect with the original mask.

An appearance score cannot establish rock height or metric traversability.
This module is deliberately outside the production backend until all gates pass.
"""
import numpy as np

def guard_obstacles(baseline_mask,navigation_probability,appearance_probability):
    if baseline_mask.shape!=navigation_probability.shape[1:] or baseline_mask.shape!=appearance_probability.shape[1:]:
        raise ValueError('Probabilities and original mask must have identical image coordinates')
    if navigation_probability.shape[0]!=4 or appearance_probability.shape[0]!=2:raise ValueError('Wrong model taxonomy')
    suppress=baseline_mask&(navigation_probability[1]<.95)&(appearance_probability[1]<.1)
    guarded=baseline_mask&~suppress
    assert not (guarded&~baseline_mask).any()
    return guarded,suppress
