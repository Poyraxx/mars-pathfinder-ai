"""Inference must preserve native detail without inventing stereo resolution."""
import io
import numpy as np
import pytest
from PIL import Image
from backend import perception
from backend.surface_inference import predict_probabilities


@pytest.fixture(autouse=True)
def isolated_navigation_session(monkeypatch):
    # The installed navigation.onnx is cached process-wide by other tests; these fixtures
    # exercise mocked sessions, so the real model must not leak in. Tests may override.
    monkeypatch.setattr(perception,'get_navigation_session',lambda:None)


def test_native_source_reaches_network_but_mask_stays_on_xyz_grid(tmp_path,monkeypatch):
    native=np.indices((8,8)).sum(0)%2*255
    Image.fromarray(native.astype('uint8')).convert('RGB').save(tmp_path/'native.png')
    Image.new('RGB',(2,2),(127,127,127)).save(tmp_path/'reference.png')
    frame={'id':'fixture','image_path':'reference.png','display_path':'native.png','native_shape':[8,8]}
    captured={}
    class Input:
        name='image';shape=[1,3,8,8]
    class Session:
        def get_inputs(self):return [Input()]
        def run(self,_,inputs):
            captured['x']=inputs['image'];logits=np.zeros((1,4,8,8),np.float32);logits[:,1]=4
            return [logits]
    monkeypatch.setattr(perception,'DATA_DIR',tmp_path)
    monkeypatch.setattr(perception,'frame_catalog',lambda:[frame])
    monkeypatch.setattr(perception,'model_status',lambda:{'input_size':320})
    monkeypatch.setattr(perception,'get_session',lambda:Session())
    _,mask,confidence,png,stats=perception.infer_navigation('fixture')
    assert captured['x'].shape==(1,3,8,8) and captured['x'].std()>1
    assert mask.shape==confidence.shape==(2,2) and (mask==1).all()
    assert Image.open(io.BytesIO(png)).size==(2,2)
    assert stats['source_shape']==stats['inference_shape']==[8,8]
    assert stats['geometry_shape']==[2,2] and stats['native_source_used']


def test_mismatched_native_camera_dimensions_are_rejected(tmp_path,monkeypatch):
    Image.new('RGB',(8,8)).save(tmp_path/'native.png');Image.new('RGB',(2,2)).save(tmp_path/'reference.png')
    monkeypatch.setattr(perception,'DATA_DIR',tmp_path)
    with pytest.raises(ValueError,match='camera dimensions'):
        perception.inference_source({'image_path':'reference.png','display_path':'native.png','native_shape':[9,8]})


def test_archive_without_native_product_keeps_reference_fallback(tmp_path,monkeypatch):
    Image.new('RGB',(256,256)).save(tmp_path/'reference.png');monkeypatch.setattr(perception,'DATA_DIR',tmp_path)
    source,reference=perception.inference_source({'image_path':'reference.png'})
    assert source.size==reference.size==(256,256)


def test_native_tiles_preserve_small_feature_at_its_actual_image_position():
    class Input:
        name='image';shape=[1,3,4,4]
    class Session:
        def get_inputs(self):return [Input()]
        def run(self,_,inputs):
            image=inputs['image'][0,0]*.229+.485
            logits=np.zeros((1,4,4,4),np.float32);logits[0,3]=np.where(image>.8,6.,-6.)
            return [logits]
    source=np.zeros((8,8,3),'uint8');source[5,5]=255;source=Image.fromarray(source)
    whole,passes=predict_probabilities(Session(),source,'whole')
    detailed,count=predict_probabilities(Session(),source,'native-tiled-v1')
    assert passes==1 and count==5
    assert whole[3].max()<.1
    assert detailed.shape==(4,8,8) and detailed.argmax(0)[5,5]==3
    assert (detailed.argmax(0)==3).sum()==1
    assert np.allclose(detailed.sum(0),1.,atol=1e-6)


def test_separate_operational_head_marks_large_block_and_excludes_self_hardware(tmp_path,monkeypatch):
    Image.new('RGB',(16,16),(120,120,120)).save(tmp_path/'native.png')
    Image.new('RGB',(8,8),(120,120,120)).save(tmp_path/'reference.png')
    frame={'id':'fixture','image_path':'reference.png','display_path':'native.png','native_shape':[16,16]}
    class Input:
        name='image';shape=[1,3,16,16]
    class Session:
        def get_inputs(self):return [Input()]
        def run(self,_,inputs):
            terrain=np.zeros((1,4,16,16),'float32');terrain[:,1]=5
            navigation=np.zeros((1,3,16,16),'float32');navigation[:,0]=5
            navigation[:,0,2:8,2:8]=0;navigation[:,2,2:8,2:8]=10
            navigation[:,0,10:15,10:15]=0;navigation[:,1,10:15,10:15]=10
            return [terrain,navigation]
    monkeypatch.setattr(perception,'DATA_DIR',tmp_path)
    monkeypatch.setattr(perception,'frame_catalog',lambda:[frame])
    monkeypatch.setattr(perception,'model_status',lambda:{})
    monkeypatch.setattr(perception,'get_session',lambda:Session())
    _,semantic,_,_,_=perception.infer('fixture')
    _,navigation,confidence,png,stats=perception.infer_navigation('fixture')
    assert semantic[6,6]==1 and navigation[6,6]==3
    assert navigation[2,2]==255 and confidence[2,2]==0
    assert np.asarray(Image.open(io.BytesIO(png)))[2,2].tolist()==[120,120,120]
    assert stats['navigation_head_used'] and stats['hardware_masked_fraction']>0


def test_sky_is_nonterrain_and_is_not_counted_as_rover_equipment(tmp_path,monkeypatch):
    Image.new('RGB',(8,8),(170,170,170)).save(tmp_path/'reference.png')
    frame={'id':'sky-fixture','image_path':'reference.png'}
    class Input:
        name='image';shape=[1,3,8,8]
    class Session:
        def get_inputs(self):return [Input()]
        def run(self,_,inputs):
            terrain=np.zeros((1,4,8,8),'float32');terrain[:,0]=4
            navigation=np.zeros((1,4,8,8),'float32');navigation[:,3]=10
            return [terrain,navigation]
    monkeypatch.setattr(perception,'DATA_DIR',tmp_path);monkeypatch.setattr(perception,'frame_catalog',lambda:[frame])
    monkeypatch.setattr(perception,'model_status',lambda:{});monkeypatch.setattr(perception,'get_session',lambda:Session())
    _,mask,confidence,png,stats=perception.infer_navigation('sky-fixture')
    assert (mask==255).all() and not confidence.any()
    assert stats['hardware_masked_fraction']==0 and stats['nonterrain_masked_fraction']==1
    assert (np.asarray(Image.open(io.BytesIO(png)))==170).all()


def test_local_only_model_class_numbers_cannot_be_confused_with_terrain(tmp_path,monkeypatch):
    Image.new('RGB',(8,8),(120,120,120)).save(tmp_path/'reference.png')
    frame={'id':'local-only','image_path':'reference.png'}
    class Input:
        name='image';shape=[1,3,8,8]
    class Session:
        def get_inputs(self):return [Input()]
        def run(self,_,inputs):
            logits=np.zeros((1,4,8,8),'float32');logits[:,0]=8
            logits[:,0,:2]=0;logits[:,3,:2]=10  # operational sky != big rock
            logits[:,0,4:6,4:6]=0;logits[:,1,4:6,4:6]=10
            logits[:,0,6:8,:2]=0;logits[:,2,6:8,:2]=10
            return [logits]
    monkeypatch.setattr(perception,'DATA_DIR',tmp_path);monkeypatch.setattr(perception,'frame_catalog',lambda:[frame])
    monkeypatch.setattr(perception,'get_navigation_session',lambda:Session())
    monkeypatch.setattr(perception,'get_session',lambda:pytest.fail('Local inference must not load global terrain network'))
    _,mask,_,_,stats=perception.infer_navigation('local-only')
    assert (mask[:2]==255).all() and mask[4,4]==3 and mask[3,5]==0
    assert mask[7,0]==255 and stats['navigation_head_used']
