"""Separate navigation obstacle/self-hardware head on a frozen terrain encoder.

Terrain taxonomy (bedrock, soil, sand, big_rock) is not traversability. A raised
bedrock block can be an obstacle and a pebble can remain non-blocking ground.
"""
from copy import deepcopy
from torch import nn
from torch.nn import functional as F
from train_rock_refinement import make_model


class SpatialRefinement(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.depthwise=nn.Conv2d(channels,channels,3,padding=1,groups=channels,bias=False)
        self.norm=nn.BatchNorm2d(channels)
        self.pointwise=nn.Conv2d(channels,channels,1,bias=False)
        nn.init.zeros_(self.pointwise.weight)
    def forward(self,x):return x+self.pointwise(F.gelu(self.norm(self.depthwise(x))))


class OperationalSurface(nn.Module):
    def __init__(self, base=None, navigation_classes=4, spatial_refinement=False, separate_navigation_encoder=False):
        super().__init__()
        self.base=base if base is not None else make_model('segformer-b0')
        self.navigation_head=deepcopy(self.base.network.decode_head)
        self.navigation_encoder=deepcopy(self.base.network.segformer) if separate_navigation_encoder else None
        previous=self.navigation_head.classifier
        self.navigation_head.classifier=nn.Conv2d(previous.in_channels,navigation_classes,1)
        if spatial_refinement:
            self.navigation_head.classifier=nn.Sequential(SpatialRefinement(previous.in_channels),self.navigation_head.classifier)
        for parameter in self.base.parameters():parameter.requires_grad=False

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self,image):
        hidden=self.base.network.segformer(pixel_values=image,output_hidden_states=True,return_dict=True).hidden_states
        terrain=self.base.network.decode_head(hidden)
        navigation_hidden=self.navigation_encoder(pixel_values=image,output_hidden_states=True,return_dict=True).hidden_states if self.navigation_encoder is not None else hidden
        navigation=self.navigation_head(navigation_hidden)
        size=image.shape[-2:]
        return (F.interpolate(terrain,size=size,mode='bilinear',align_corners=False),
                F.interpolate(navigation,size=size,mode='bilinear',align_corners=False))
