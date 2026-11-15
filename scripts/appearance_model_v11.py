"""A separately trainable appearance decoder on an immutable navigation encoder."""
from copy import deepcopy
from torch import nn
from torch.nn import functional as F
from operational_model import OperationalSurface

def load_navigation(path):
    import torch
    model=OperationalSurface(spatial_refinement=True,separate_navigation_encoder=True)
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));return model.eval()

class RockAppearance(nn.Module):
    def __init__(self,navigation):
        super().__init__();self.encoder=deepcopy(navigation.navigation_encoder);self.head=deepcopy(navigation.navigation_head)
        old=self.head.classifier[1];new=nn.Conv2d(old.in_channels,2,1)
        new.weight.data.copy_(old.weight.data[:2]);new.bias.data.copy_(old.bias.data[:2]);self.head.classifier[1]=new
        for p in self.encoder.parameters():p.requires_grad=False
    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self
    def forward(self,x):
        # No old navigation or terrain tensor belongs to this optimizer.
        hidden=self.encoder(pixel_values=x,output_hidden_states=True,return_dict=True).hidden_states
        return F.interpolate(self.head(hidden),size=x.shape[-2:],mode='bilinear',align_corners=False)
