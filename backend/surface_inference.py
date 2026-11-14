"""Shared native-pixel inference for runtime and honest candidate evaluation."""
import numpy as np
import threading
from PIL import Image
from scipy import ndimage

_inference_guard=threading.Lock()


def navigation_masks(probability,native_size,obstacle_threshold=.7,hardware_threshold=.8,hardware_margin_px=3):
    """High-confidence self exclusion; its boundary becomes UNKNOWN ground."""
    classes=probability.argmax(0)
    hardware=(classes==2)&(probability[2]>=hardware_threshold)
    if hardware.any() and hardware_margin_px:
        margin=max(1,int(round(hardware_margin_px*probability.shape[-1]/native_size[0])))
        hardware=ndimage.binary_dilation(hardware,iterations=margin)
    obstacle=(classes==1)&(probability[1]>=obstacle_threshold)&~hardware
    return obstacle,hardware


def _run(session,image,size):
    pixels=np.asarray(image.resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)/255
    tensor=((pixels-[.485,.456,.406])/[.229,.224,.225]).transpose(2,0,1)[None].astype('float32')
    outputs=session.run(None,{session.get_inputs()[0].name:tensor})
    probabilities=[]
    for output in outputs:
        logits=output[0];logits-=logits.max(0,keepdims=True)
        probability=np.exp(logits);probability/=probability.sum(0,keepdims=True)
        probabilities.append(probability)
    return probabilities


def _starts(length,side):
    if length<=side:return [0]
    if length<=side*2:return [0,length-side]
    positions=list(range(0,length-side+1,max(1,side*3//4)))
    if positions[-1]!=length-side:positions.append(length-side)
    return positions


def predict_probabilities(session,source,policy='whole'):
    outputs,passes=predict_outputs(session,source,policy)
    return outputs[0],passes


def predict_outputs(session,source,policy='whole'):
    # Startup replay warming and an interactive frame may arrive together.
    # Share one ONNX workspace instead of multiplying peak memory on free hosts.
    with _inference_guard:
        return _predict_outputs(session,source,policy)


def _predict_outputs(session,source,policy):
    if policy not in ('whole','native-tiled-v1'):raise ValueError('Unknown surface inference policy')
    size=session.get_inputs()[0].shape[-1]
    if not isinstance(size,int):raise ValueError('Surface model needs a fixed square input size')
    probabilities=_run(session,source,size)
    if policy=='whole' or min(source.size)<=size:return probabilities,1
    # The whole-image pass retains context. Additional REAL native windows give
    # small objects more feature pixels; no image content or depth is generated.
    width,height=source.size
    if width>2048 or height>2048:raise ValueError('Native tiled inference exceeds bounded source size')
    accumulators=[np.stack([np.asarray(Image.fromarray(p).resize(source.size,Image.Resampling.BILINEAR)) for p in probability]) for probability in probabilities]
    denominator=np.ones((height,width),np.float32)
    one=np.maximum(.2,np.hanning(size).astype(np.float32))
    weight=2*np.outer(one,one);passes=1
    for top in _starts(height,size):
        for left in _starts(width,size):
            tile=source.crop((left,top,left+size,top+size))
            predictions=_run(session,tile,size)
            for accumulator,prediction in zip(accumulators,predictions):
                accumulator[:,top:top+size,left:left+size]+=prediction*weight
            denominator[top:top+size,left:left+size]+=weight;passes+=1
    for accumulator in accumulators:accumulator/=denominator
    return accumulators,passes
