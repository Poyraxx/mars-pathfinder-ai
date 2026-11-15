"""A destination flag must use the actual endpoint's camera/occlusion test."""
import numpy as np
from backend import camera_route


def projection(monkeypatch, goal, occluded=False,nonterrain=False):
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
    excluded=np.zeros((100,100),bool)
    if nonterrain:excluded[48:53,68:74]=True
    monkeypatch.setattr(camera_route,'visibility',lambda _:(excluded,None))
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


def test_equipment_without_stereo_depth_still_hides_goal_and_route(monkeypatch):
    result=projection(monkeypatch,4.2,nonterrain=True)
    assert result['destination'] is None
    assert result['destination_visibility']=='nonterrain_occluded'
    assert all(not(.68<=p[0]<.74 and .48<=p[1]<.53) for s in result['segments'] for p in s)


def test_ground_height_support_never_uses_masked_hardware(tmp_path,monkeypatch):
    xyz=np.array([[[1.,2.,-2.],[1.,2.,0.],[2.,3.,0.]]])
    np.save(tmp_path/'xyz.npy',xyz);monkeypatch.setattr(camera_route,'DATA_DIR',tmp_path)
    frame={'xyz_path':'xyz.npy','places':{'archive_site_to_global_ned_rotation':np.eye(3),
                                        'archive_site_to_global_ned_translation_m':[0,0,0]}}
    ground,rows,cols,shape,ned=camera_route.ground_pixels(frame,{'radius_m':camera_route.R},np.array([[True,False,False]]))
    assert cols.tolist()==[1,2] and np.all(ned[:,2]==0)


def test_no_visible_ground_does_not_assert_a_goal(monkeypatch):
    monkeypatch.setattr(camera_route,'ground_pixels',lambda *_:(np.empty((0,2)),[],[],(10,10),np.empty((0,3))))
    monkeypatch.setattr(camera_route,'visibility',lambda _:(None,None))
    result=camera_route.project_route({'id':'empty'}, {},[(0,0)],.5,destination_xy=[0,0])
    assert result['segments']==[] and result['destination'] is None
