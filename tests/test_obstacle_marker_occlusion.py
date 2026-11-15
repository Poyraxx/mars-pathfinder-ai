"""Remembered rocks remain in world memory but cannot be drawn on the arm."""
import numpy as np
from backend import local_navigation as local,camera_visibility

def fixture(monkeypatch):
    frame={'camera_model':{'native_shape':[100,100]}}
    terrain={'height':np.zeros((2,2))}
    monkeypatch.setattr(local,'frame_basis',lambda _:(np.zeros(3),np.array([1.,0.]),np.array([0.,1.])))
    monkeypatch.setattr(local,'_centres',lambda cells,_:np.zeros((len(cells),2)))
    monkeypatch.setattr(local,'site_points_to_pixels',lambda *_:np.array([[25.,25.],[75.,75.],[np.nan,0.],[100.,50.]]))
    return frame,terrain,np.array([[0,0],[0,1],[1,0],[1,1]])

def test_marker_behind_hardware_is_hidden_without_removing_world_track(monkeypatch):
    frame,terrain,cells=fixture(monkeypatch);original=cells.copy();excluded=np.zeros((10,10),bool);excluded[2,2]=True
    monkeypatch.setattr(camera_visibility,'visibility',lambda _:(excluded,None))
    assert local.project_cells(frame,terrain,cells)==[[.75,.75]]
    assert np.array_equal(cells,original)

def test_clear_pixels_keep_markers_and_outside_pixels_are_never_indexed(monkeypatch):
    frame,terrain,cells=fixture(monkeypatch)
    monkeypatch.setattr(camera_visibility,'visibility',lambda _:(np.zeros((10,10),bool),None))
    assert local.project_cells(frame,terrain,cells)==[[.25,.25],[.75,.75]]

def test_missing_image_mask_preserves_existing_projection(monkeypatch):
    frame,terrain,cells=fixture(monkeypatch)
    monkeypatch.setattr(camera_visibility,'visibility',lambda _:(None,None))
    assert local.project_cells(frame,terrain,cells)==[[.25,.25],[.75,.75]]
