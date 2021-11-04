#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Model itself

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from compressai.models import MeanScaleHyperprior
from compressai.layers import GDN, ResidualBlockWithStride, subpel_conv3x3
from compressai.models import Cheng2020Anchor

from .endecoder import ME_Spynet, flow_warp, Warp_net
from .functions import conv, deconv


# In[ ]:


device = torch.device("cuda")


# In[ ]:


class MyMeanScaleHyperprior(MeanScaleHyperprior):
    def __init__(self, in_channel, N=128, M=128):
        super(MyMeanScaleHyperprior, self).__init__(N=N, M=M)
        
        self.g_a = Cheng2020Anchor(N=N).g_a
        self.g_a[0] = ResidualBlockWithStride(in_channel, N, stride=2)

        self.out_channel = in_channel
        
        self.g_s = Cheng2020Anchor(N=N).g_s
        self.g_s[-1] = subpel_conv3x3(in_ch=N, out_ch=in_channel, r=2)
        
        self.h_a = Cheng2020Anchor(N=N).h_a
        
        self.h_s = Cheng2020Anchor(N=N).h_s
        
    def forward(self, x):
        return super(MyMeanScaleHyperprior, self).forward(x)
        

class P_Model(nn.Module):
    def __init__(self, N=128, M=128):
        super(P_Model, self).__init__()
        
        # From pretrained
        self.opticFlow = ME_Spynet()
        self.warpnet = Warp_net()

        self.flow_model = MyMeanScaleHyperprior(in_channel=2, N=N, M=M)
        self.res_model = MyMeanScaleHyperprior(in_channel=3, N=N, M=M)
        
        
    def motioncompensation(self, ref, mv):
        warpframe = flow_warp(ref, mv)
        inputfeature = torch.cat((warpframe, ref), 1)
        prediction = self.warpnet(inputfeature) + warpframe
        return prediction, warpframe


    def forward(self, x_previous, x_current, train):
        N, C, H, W = x_current.size()
        num_pixels = N * H * W

        # Encode & decode the flow
        mv_p2c = self.opticFlow(x_current, x_previous)
        
        flow_enc_result = self.flow_model(mv_p2c)
        flow_hat = flow_enc_result["x_hat"]
        flow_likelihoods = flow_enc_result["likelihoods"]

        # Apply motion compensation and post processing
        prediction, warpframe = self.motioncompensation(x_previous, flow_hat)
        
        residual = x_current - prediction

        # Encode & decode residual
        res_enc_result = self.res_model(residual)
        res_hat = res_enc_result["x_hat"]
        res_likelihoods = res_enc_result["likelihoods"]

        x_current_hat = res_hat + prediction

        # Calculate the rate and size
        size_flow = sum(
            (torch.log(likelihoods).sum() / (-math.log(2))) for likelihoods in flow_likelihoods.values()
        )
        
        rate_flow = size_flow / num_pixels

        size_residual = sum(
            (torch.log(likelihoods).sum() / (-math.log(2))) for likelihoods in res_likelihoods.values()
        )
        
        rate_residual = size_residual / num_pixels
        
        if train:
            return x_current_hat, (rate_flow + rate_residual) / 2.0
        else:
            return x_current_hat, (rate_flow + rate_residual) / 2.0, size_flow + size_residual, size_flow / size_flow + size_residual