"""Execute the exact shared web/video geometry on controlled projected paths."""
import numpy as np
from scripts.export_replay_video import overlay_geometries


def fixture():
    return {'ribbons':[[[.45,.88],[.55,.88],[.52,.7],[.48,.7]],
                       [[.48,.7],[.52,.7],[.506,.61],[.494,.61]]],
            'groundM':[[2.,3.],[3.,4.]],'localLength':4.,
            'segments':[[[.49,.54,5.],[.46,.48,8.],[.42,.4,14.],[.36,.3,25.]]]}


def test_near_and_far_share_one_outline_with_a_gradual_taper():
    result=overlay_geometries([fixture()])[0]
    assert result['hasLocal'] and len(result['strokes'])==1
    stroke=result['strokes'][0];centre=np.asarray(stroke['centre']);width=np.asarray(stroke['width'])
    assert np.isfinite(centre).all() and np.isfinite(width).all()
    assert len(stroke['outline'])==2*len(centre)
    assert centre[0].tolist()==[.5,.985] and result['end']==[.36,.3]
    assert width[-1]<width[0]/5 and np.abs(np.diff(width)).max()<.01
    join=np.where(np.linalg.norm(centre-[.5,.61],axis=1)<1e-7)[0][-1]
    assert abs(width[join+1]-width[join])<.0001
    assert stroke['planned'][0]==centre[join].tolist()


def test_map_context_survives_without_a_measured_local_route():
    frame=fixture();frame['ribbons']=[];frame['groundM']=[]
    result=overlay_geometries([frame])[0]
    assert not result['hasLocal'] and result['strokes']
    stroke=result['strokes'][0]
    assert stroke['centre'][0]==frame['segments'][0][0][:2]
    assert stroke['planned']==stroke['centre']
    assert [.5,.985] not in stroke['centre']


def test_camera_exit_is_not_filled_by_a_solid_shortcut():
    frame=fixture();frame['segments'].append([[.95,.75,30.],[.97,.7,33.],[.99,.65,36.]])
    result=overlay_geometries([frame])[0]
    assert len(result['strokes'])==2 and result['hidden']==[]
    assert overlay_geometries([{'ribbons':[],'segments':[]}])[0]['strokes']==[]


def test_centre_guide_link_is_entirely_planned_when_no_local_route_exists():
    frame=fixture();frame.update(ribbons=[],groundM=[],planningLink=True)
    result=overlay_geometries([frame])[0]
    assert not result['hasLocal']
    assert result['strokes'][0]['centre'][0]==[.5,.985]
    assert result['strokes'][0]['planned']==result['strokes'][0]['centre']


def test_taper_is_distributed_through_the_middle_instead_of_pinching_at_the_end():
    frame={'ribbons':[[[.45,.85],[.55,.85],[.55,.35],[.45,.35]]],
           'groundM':[[2.,6.]],'localLength':6.,
           'segments':[[[.5,.30,7.],[.5,.25,15.],[.5,.2,40.]]]}
    stroke=overlay_geometries([frame])[0]['strokes'][0]
    c=np.array(stroke['centre']);w=np.array(stroke['width'])
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(c,axis=0),axis=1))]
    join=np.argmin(np.linalg.norm(c-[.5,.35],axis=1))
    middle=np.argmin(abs(arc-arc[join]*.5))
    assert .3*w[0]<w[middle]<.8*w[0]
    for i in range(join):
        j=min(join,np.searchsorted(arc,arc[i]+.05))
        assert w[i]/w[j]<2.8


def test_uturn_swing_tapers_without_inventing_a_visible_return_route():
    frame=fixture();frame.update(segments=[],manoeuvre=True)
    result=overlay_geometries([frame])[0];stroke=result['strokes'][0]
    assert result['hasLocal'] and len(result['strokes'])==1
    assert stroke['width'][-1]<stroke['width'][0]/8
    assert stroke['planned']==[] and result['end']==[.5,.61]


def test_short_near_destination_keeps_unverified_map_context():
    frame={'segments':[[[.4,.8,0.],[.45,.65,.5],[.5,.5,1.2]]],'planningLink':True}
    result=overlay_geometries([frame])[0]
    assert not result['hasLocal'] and result['strokes']
    assert result['end']==[.5,.5] and result['strokes'][0]['centre'][0]==[.5,.985]
    assert result['strokes'][0]['planned']==result['strokes'][0]['centre']


def test_unmeasured_centre_link_has_one_bearing_instead_of_an_opposite_side_sweep():
    frame={'segments':[[[.46,.6,2.],[.52,.5,3.],[.6,.4,4.]]],'planningLink':True}
    stroke=overlay_geometries([frame])[0]['strokes'][0]
    c=np.asarray(stroke['centre'])[:33]
    assert np.all(np.diff(c[:,0])<=1e-9) and np.all(np.diff(c[:,1])<0)
    assert np.max(abs((c[:,0]-.5)/(.46-.5)-(c[:,1]-.985)/(.6-.985)))<1e-7


def test_measured_approach_link_does_not_overshoot_its_first_ground_point():
    frame={'ribbons':[[[.36,.65],[.44,.65],[.54,.55],[.46,.55]]],
           'groundM':[[2.,3.]],'segments':[]}
    stroke=overlay_geometries([frame])[0]['strokes'][0]
    c=np.asarray(stroke['centre'])
    assert c[:,0].min()>=.4-1e-7
    assert c[0].tolist()==[.5,.985]
