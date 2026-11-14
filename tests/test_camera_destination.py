"""A destination flag must use the actual endpoint's camera/occlusion test."""
import numpy as np
from backend import camera_route


def projection(monkeypatch, goal, occluded=False):
    # Analytic geometry fixture: NED ground plane, camera one metre above it.
    model={'validation':{'accepted':True},'native_shape':[100,100],'C':[0,0,-1],
           'rover_to_site_rotation':np.eye(3).tolist(),'rover_origin_in_site_m':[0,0,0]}
    places={'archive_site_to_global_ned_translation_m':[0,0,0],
            'archive_site_to_global_ned_rotation':np.eye(3).tolist(),
            'corrected_rover_origin_global_ned_m':[0,0,0]}
    frame={'id':'analytic','xyz_path':'fixture.npy','places':places,'camera_model':model}
    x=np.arange(.5,21,.5);ground=np.column_stack((x,np.zeros_like(x)))
    ned=np.column_stack((np.zeros_like(x),x,np.zeros_like(x)))
    monkeypatch.setattr(camera_route,'ground_pixels',lambda *_:(ground,np.zeros(len(x),int),np.zeros(len(x),int),(100,100),ned))
    monkeypatch.setattr(camera_route,'site_points_to_pixels',lambda pts,_:np.column_stack((50+pts[:,1]*5,50+pts[:,0]*5)))
    xyz=np.zeros((100,100,3));xyz[:,:,1]=1 if occluded else 0
    monkeypatch.setattr(camera_route.np,'load',lambda *_:xyz)
    meta={'id':'analytic-no-dem','cell_m':1.,'west_m':0.,'north_m':.5}
    return camera_route.project_route(frame,meta,[(0,c) for c in range(21)],.5,destination_xy=[goal,0])


def test_visible_flag_projects_the_exact_requested_goal(monkeypatch):
    result=projection(monkeypatch,4.2)
    assert result['destination']==[.71,.5]
    assert result['segments'][-1][-1][:2]==[.71,.5]


def test_last_visible_fragment_is_not_a_destination_when_goal_is_out_of_view(monkeypatch):
    result=projection(monkeypatch,20.)
    assert result['segments'] and result['destination'] is None


def test_nearer_stereo_surface_hides_the_destination_flag(monkeypatch):
    assert projection(monkeypatch,4.2,occluded=True)['destination'] is None
