# Copyright 2025 GT Synergy Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np 
from collections import OrderedDict
import os
# import vim_GEMM

REAL_INT8 = True

class RoundQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):    
        return input.round()
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class GradientScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n_lv, size):
        ctx.save_for_backward(torch.Tensor([n_lv, size]))
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        saved, = ctx.saved_tensors
        n_lv, size = int(saved[0]), float(saved[1])

        if n_lv == 0:
            return grad_output, None, None
        else:
            scale = 1 / np.sqrt(n_lv * size)
            return grad_output.mul(scale), None, None

class Q_Linear(nn.Linear):
    """
    Quantized Linear layer with tunable quantization parameters ('s').
    This layer supports per-channel quantization and can be initialized with a specific number of levels.
    Specific Method: Symmetric Uniform Per-Channel Quantization
    """
    def __init__(self, *args, act_func=None, **kargs):
        super(Q_Linear, self).__init__(*args, **kargs)
        self.act_func = act_func
        self.n_lv = 16
        self.qmax = self.n_lv // 2 - 1 
        self.qmin = -self.qmax
        self.per_channel = False
        self.s = None
        self.num = 100
        self.eps = torch.tensor(1e-8)
        self.smoothing = False
        self.real_int8 = False
        self.qbit = 4
        self.int_weight = torch.Tensor(1)
        
    def quantize_efficient(self, x_round, scale, zero=0):
        q = torch.clamp(x_round + zero, self.qmin , self.qmax)
        return scale * (q - zero)
    
    def lp_loss(self, pred, tgt, p=2.4):
        x = (pred - tgt).abs().pow(p)
        if not self.per_channel:
            return x.mean()
        else:
            y = torch.flatten(x, 1)
            return y.mean(1)

    def set_bits(self, n_lv):
        self.n_lv = n_lv
        self.qmax = n_lv // 2 - 1
        self.qmin = -self.qmax
    
    def set_real_int8(self):
        self.real_int8 = True
        if self.n_lv == 256:
            self.int_weight = torch.tensor(self.weight.to(torch.int8))
            self.qbit = 8
        elif self.n_lv == 16:
            self.int_weight = torch.tensor(self.weight.to(torch.int8))[:, :self.weight.shape[1]//2]
            self.qbit = 8

    def _weight_quant(self): 
        s = self.s 
        weight = self.weight
        weight = F.hardtanh((weight / s), self.qmin, self.qmax)
        weight = RoundQuant.apply(weight) * s
        return weight

    def _weight_int(self):
        with torch.no_grad():        
            weight = F.hardtanh(self.weight / self.s,- (self.n_lv // 2 - 1), self.n_lv // 2 - 1)
            weight = torch.round(weight)
        return weight
        
    def forward(self, x):
        if self.act_func is not None:
            x = self.act_func(x)
        
        del self.s
        self.qmax = self.n_lv // 2 - 1
        self.qmin = -self.qmax  
        weight_copy = self.weight.clone()
        weight_copy = weight_copy.flatten(1) 

        xmax = weight_copy.max(1)[0]
        max_val = torch.max(xmax, torch.zeros_like(xmax))
        val = torch.max(max_val / self.qmax, self.eps).unsqueeze(1)
        self.register_parameter("s",torch.nn.Parameter(val))
        
        try:
            weight = self._weight_quant()
        except:
            breakpoint()
        return F.linear(x, weight, self.bias)

class Q_Act(nn.Module):
    def __init__(self):
        super(Q_Act, self).__init__()
        self.n_lv = 16
        self.alpha = 2.8
        self.qmax = self.n_lv//2 - 1
        self.qmin = -1*self.qmax
        self.per_channel = False
        self.s = None
        self.num = 100
        self.eps = torch.tensor(1e-8)
        self.smoothing = False
        self.real_int8 = False
        self.n_lva_o = 2**8
        
    def quantize_efficient(self, x_round, scale, zero=0):
        q = torch.clamp(x_round + zero, self.qmin , self.qmax)
        return scale * (q - zero)
        
    def log_quantize_efficient(self, x_abs, x_sign, scale):
        # y = clip(round(-log_2(x_abs)))
        # return scale * 2^(-y) * sign(x)
        y = torch.clamp((-1* torch.log2(x_abs)).round(), self.qmin, self.qmax)
        return scale * 2**(-1*y) * x_sign
     
    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        if not self.per_token:
            return x.mean()
        else:
            y = torch.flatten(x, 1)
            return y.mean(1)
        
    def set_bits(self, n_lv):
        self.n_lv = n_lv
        self.qmax = n_lv // 2 - 1
        self.qmin = -self.qmax
    
    def set_real_int8(self):
        self.real_int8 = True 
    
    def extracter_refresh(self, x, n_refresh=10):
        '''
            Extract the outlier channel using list, update new outliers, refresh every n_refresh time steps
            1. Obtain mean, std per batch, per group, per time steps and calculate threshold
            2. Extract outliers and inliers based on the threshold
        '''
        # reshape input tensor
        B, L, D = x.shape                                                   # B, L, D
        concat = False
        pad = 0
        new_L = L
        if L%n_refresh != 0:
            new_L = (L//n_refresh +1) * n_refresh
            pad = new_L - L
            x = torch.cat([x,torch.zeros(B,pad,D, device=x.device)], dim=1)
            concat = True
        x = x.view(B, new_L//n_refresh, n_refresh, D)                       # B, L/10, 10, D

        # find threshold for each timesteps
        mean = torch.mean(torch.abs(x), dim=3, keepdim=True)                # B, L/10, 10, 1
        std = torch.std(torch.abs(x), dim=3, keepdim=True)                  # B, L/10, 10, 1
        threshold = mean + self.alpha*std
        
        # find outliers per batch, per n_refresh, per timestep
        outlier_mask = (torch.abs(x) > threshold)                           # B, L/10, 10, D
        # cumulate for each timestep to apply outlier list push
        outlier_mask = outlier_mask.cumsum(dim=2).bool()                    # B, L/10, 10, D
        inlier_mask = ~outlier_mask
        
        # extract the out/inliers
        outlier_x = torch.where(outlier_mask, x, torch.zeros_like(x))       # (B, L/10, 10, D)
        inlier_x = torch.where(inlier_mask, x, torch.zeros_like(x))         # (B, L/10, 10 , D)
        
        # reshape to B, L, D
        outlier_x = outlier_x.view(B,new_L, D)
        inlier_x = inlier_x.view(B, new_L, D)
        outlier_mask = outlier_mask.view(B, new_L, D)
        inlier_mask = inlier_mask.view(B, new_L, D)
        if concat:
            outlier_x    = outlier_x[:,:L,:]
            inlier_x     = inlier_x[:,:L,:]     
            outlier_mask = outlier_mask[:,:L,:]       
            inlier_mask  = inlier_mask[:,:L,:]       
        return outlier_x, outlier_mask, inlier_x, inlier_mask           

    def extracter_requant(self, x, n_refresh=10, static_o_list=None, static_s=None, qtype = 'uni'):
        '''
            Extract the outlier channel using list, update new outliers if detected, refresh every n_refresh time steps
        '''
        # reshape input tensor
        ###############################################################
        B, L, D = x.shape                                                   # B, L, D
        concat = False
        pad = 0
        new_L = L
        o_qmax = self.n_lva_o//2 - 1
        o_qmin = -1*o_qmax
        
        if L%n_refresh != 0:
            new_L = (L//n_refresh +1) * n_refresh
            pad = new_L - L
            x = torch.cat([x,torch.zeros(B,pad,D,device=x.device)], dim=1)
            static_s = static_s.view(L)
            static_s = torch.cat([static_s,torch.ones(pad,device=static_s.device)], dim=0)    # L -> new_L
            concat = True

        x = x.view(B, new_L//n_refresh, n_refresh, D)                       # B, L/10, 10, D
        static_s= static_s.view(1, new_L//n_refresh, n_refresh)             # 1, L/10, 10
        o_list = torch.zeros(B, new_L//n_refresh, D, device = x.device)     # B, L/10, D
        ###############################################################

        for i in range(n_refresh):
            # exclude outliers in list
            inlier_x = x[:,:,i,:] * (1-o_list)                              # B, L/10, D
            
            # find scale for inliers
            if qtype == 'uni':
                dynamic_s = torch.abs(inlier_x).max(2)[0]/self.qmax         # B, L/10
            elif qtype == 'non-uni':
                dynamic_s = torch.abs(inlier_x).max(2)[0]                   # B, L/10
            else:
                assert False
                
            # new outlier detection, exclusion
            if (dynamic_s > static_s[:,:,i]).any():
                mean = torch.mean(torch.abs(inlier_x),dim=2,keepdim=True)   # B, L/10, 1
                std = torch.std(torch.abs(inlier_x),dim=2,keepdim=True)     # B, L/10, 1
                threshold = mean + self.alpha*std
                
                o_list = (o_list.bool() | (torch.abs(inlier_x) > threshold)).float()       # B, L/10, D
                inlier_x = x[:,:,i,:] * (1-o_list)                          # B, L/10, D
            
            s = static_s[:,:,i].unsqueeze(2)                                # 1, L/10, 1
            # quantize the inliers
            if qtype == 'uni':
                inlier_q = F.hardtanh((inlier_x/s).round(), self.qmin, self.qmax)
                inlier_x = inlier_q * s                                     # B, L/10, D
            elif qtype == 'non-uni':
                in_abs = torch.sqrt(inlier_x**2)
                in_sign = inlier_x / (in_abs+ 1e-15)
                in_abs = in_abs / s
                in_abs = torch.clamp(in_abs, 1e-8, 1.0)
                inlier_x = self.log_quantize_efficient(in_abs, in_sign, s)
            else : 
                assert False
            
            # quantize the outliers 
            outlier_x = x[:,:,i,:] * o_list                                 # B, L/10, D
            outlier_s = torch.abs(outlier_x).max(2)[0].clamp(min=1e-8)/o_qmax             # B, L/10
            outlier_s = outlier_s.unsqueeze(2)                              # B, L/10, 1
            outlier_q = F.hardtanh((outlier_x/outlier_s).round(), o_qmin, o_qmax)
            outlier_x = outlier_q * outlier_s
            
            x[:,:,i,:] = inlier_x + outlier_x
        x = x.view(B, new_L, D) # B, L, D
        if concat :
            x = x[:,:L,:]
        return x
 
    def forward(self, x):
        del self.s
        x_copy = x.clone()
        _, _, x_copy, _ = self.extracter_refresh(x_copy, 10)

        b,l,d = x_copy.shape
        x_copy = x_copy.permute(0,2,1) # b, d, l
        x_copy = x_copy.reshape(-1, l) # bd, l
        x_copy = x_copy.permute(1,0)   # l, bd

        xmax = torch.abs(x_copy).max(1)[0] # l,
        max_val_pos = torch.max(xmax, torch.zeros_like(xmax))
        val = torch.max(max_val_pos, self.eps).unsqueeze(1).unsqueeze(0) / self.qmax
        self.register_parameter("s",torch.nn.Parameter(val))

        x = self.extracter_requant(x, 10, None, self.s, 'uni')
        x = x.half()

        return x
    ######################################################################################################

class Q_Linear_high(nn.Linear):
    """
    Quantized Linear layer with tunable quantization parameters ('s').
    This layer supports per-channel quantization and can be initialized with a specific number of levels.
    Specific Method: Symmetric Uniform Per-Channel Quantization
    """
    def __init__(self, *args, act_func=None, **kargs):
        super(Q_Linear_high, self).__init__(*args, **kargs)
        self.act_func = act_func
        self.n_lv = 2**8
        self.qmax = self.n_lv // 2 - 1 
        self.qmin = -self.qmax
        self.per_channel = False
        self.s = None
        self.num = 100
        self.eps = torch.tensor(1e-8)
        self.smoothing = False
        self.real_int8 = False
        self.qbit = 8
        self.int_weight = torch.Tensor(1)
        
    def quantize_efficient(self, x_round, scale, zero=0):
        q = torch.clamp(x_round + zero, self.qmin , self.qmax)
        return scale * (q - zero)
    
    def lp_loss(self, pred, tgt, p=2.4):
        x = (pred - tgt).abs().pow(p)
        if not self.per_channel:
            return x.mean()
        else:
            y = torch.flatten(x, 1)
            return y.mean(1)

    def set_bits(self, n_lv):
        self.n_lv = n_lv
        self.qmax = n_lv // 2 - 1
        self.qmin = -self.qmax
    
    def set_real_int8(self):
        self.real_int8 = True
        if self.n_lv == 256:
            self.int_weight = torch.tensor(self.weight.to(torch.int8))
            self.qbit = 8
        elif self.n_lv == 16:
            self.int_weight = torch.tensor(self.weight.to(torch.int8))[:, :self.weight.shape[1]//2]
            self.qbit = 4

    def _weight_quant(self): 
        s = self.s 
        weight = self.weight
        weight = F.hardtanh((weight / s), self.qmin, self.qmax)
        weight = RoundQuant.apply(weight) * s
        return weight

    def _weight_int(self):
        with torch.no_grad():        
            weight = F.hardtanh(self.weight / self.s,- (self.n_lv // 2 - 1), self.n_lv // 2 - 1)
            weight = torch.round(weight)
        return weight
        
    def forward(self, x):
        if self.act_func is not None:
            x = self.act_func(x)
        del self.s
        self.qmax = self.n_lv // 2 - 1
        self.qmin = -self.qmax  
        weight_copy = self.weight.clone()
        weight_copy = weight_copy.flatten(1) 
        
        xmax = weight_copy.max(1)[0]
        max_val = torch.max(xmax, torch.zeros_like(xmax))
        val = torch.max(max_val / self.qmax, self.eps).unsqueeze(1)
        self.register_parameter("s",torch.nn.Parameter(val))
    
        try:
            weight = self._weight_quant()
        except:
            breakpoint()
        return F.linear(x, weight, self.bias)

class Q_Act_high(nn.Module):
    def __init__(self):
        super(Q_Act_high, self).__init__()
        self.n_lv = 2**8
        self.alpha = 2.8
        self.qmax = self.n_lv//2 - 1
        self.qmin = -1*self.qmax
        self.per_channel = False
        self.s = None
        self.num = 100
        self.eps = torch.tensor(1e-8)
        self.smoothing = False
        self.real_int8 = False
        self.n_lva_o = 2**16
        
    def quantize_efficient(self, x_round, scale, zero=0):
        q = torch.clamp(x_round + zero, self.qmin , self.qmax)
        return scale * (q - zero)
        
    def log_quantize_efficient(self, x_abs, x_sign, scale):
        # y = clip(round(-log_2(x_abs)))
        # return scale * 2^(-y) * sign(x)
        y = torch.clamp((-1* torch.log2(x_abs)).round(), self.qmin, self.qmax)
        return scale * 2**(-1*y) * x_sign
     
    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        if not self.per_token:
            return x.mean()
        else:
            y = torch.flatten(x, 1)
            return y.mean(1)
        
    def set_bits(self, n_lv):
        self.n_lv = n_lv
        self.qmax = n_lv // 2 - 1
        self.qmin = -self.qmax
    
    def set_real_int8(self):
        self.real_int8 = True 
    
    def extracter_refresh(self, x, n_refresh=10):
        '''
            Extract the outlier channel using list, update new outliers, refresh every n_refresh time steps
            1. Obtain mean, std per batch, per group, per time steps and calculate threshold
            2. Extract outliers and inliers based on the threshold
        '''
        # reshape input tensor
        B, L, D = x.shape                                               # B, L, D
        concat = False
        pad = 0
        new_L = L
        if L%n_refresh != 0:
            new_L = (L//n_refresh +1) * n_refresh
            pad = new_L - L
            x = torch.cat([x,torch.zeros(B,pad,D, device=x.device)], dim=1)
            concat = True
        x = x.view(B, new_L//n_refresh, n_refresh, D)                   # B, L/10, 10, D

        # find threshold for each timesteps
        mean = torch.mean(torch.abs(x), dim=3, keepdim=True)            # B, L/10, 10, 1
        std = torch.std(torch.abs(x), dim=3, keepdim=True)              # B, L/10, 10, 1
        threshold = mean + self.alpha*std
        
        # find outliers per batch, per n_refresh, per timestep
        outlier_mask = (torch.abs(x) > threshold)                       # B, L/10, 10, D
        # cumulate for each timestep to apply outlier list push
        outlier_mask = outlier_mask.cumsum(dim=2).bool()                # B, L/10, 10, D
        inlier_mask = ~outlier_mask
        
        # extract the out/inliers
        outlier_x = torch.where(outlier_mask, x, torch.zeros_like(x))   # (B, L/10, 10, D)
        inlier_x = torch.where(inlier_mask, x, torch.zeros_like(x))     # (B, L/10, 10 , D)
        
        # reshape to B, L, D
        outlier_x = outlier_x.view(B,new_L, D)
        inlier_x = inlier_x.view(B, new_L, D)
        outlier_mask = outlier_mask.view(B, new_L, D)
        inlier_mask = inlier_mask.view(B, new_L, D)
        if concat:
            outlier_x    = outlier_x[:,:L,:]
            inlier_x     = inlier_x[:,:L,:]     
            outlier_mask = outlier_mask[:,:L,:]       
            inlier_mask  = inlier_mask[:,:L,:]       
        return outlier_x, outlier_mask, inlier_x, inlier_mask           

    def extracter_requant(self, x, n_refresh=10, static_o_list=None, static_s=None, qtype = 'uni'):
        '''
            Extract the outlier channel using list, update new outliers if detected, refresh every n_refresh time steps
        '''
        # reshape input tensor
        ###############################################################
        B, L, D = x.shape                                                   # B, L, D
        concat = False
        pad = 0
        new_L = L
        o_qmax = self.n_lva_o//2 - 1
        o_qmin = -1*o_qmax
        
        if L%n_refresh != 0:
            new_L = (L//n_refresh +1) * n_refresh
            pad = new_L - L
            x = torch.cat([x,torch.zeros(B,pad,D,device=x.device)], dim=1)
            static_s = static_s.view(L)
            static_s = torch.cat([static_s,torch.ones(pad,device=static_s.device)], dim=0)    # L -> new_L
            concat = True

        x = x.view(B, new_L//n_refresh, n_refresh, D)                       # B, L/10, 10, D
        static_s= static_s.view(1, new_L//n_refresh, n_refresh)             # 1, L/10, 10
        o_list = torch.zeros(B, new_L//n_refresh, D, device = x.device)     # B, L/10, D
        ###############################################################

        for i in range(n_refresh):
            # exclude outliers in list
            inlier_x = x[:,:,i,:] * (1-o_list)                              # B, L/10, D
            
            # find scale for inliers
            if qtype == 'uni':
                dynamic_s = torch.abs(inlier_x).max(2)[0]/self.qmax         # B, L/10
            elif qtype == 'non-uni':
                dynamic_s = torch.abs(inlier_x).max(2)[0]                   # B, L/10
            else:
                assert False
                
            # new outlier detection, exclusion
            if (dynamic_s > static_s[:,:,i]).any():
                mean = torch.mean(torch.abs(inlier_x),dim=2,keepdim=True)   # B, L/10, 1
                std = torch.std(torch.abs(inlier_x),dim=2,keepdim=True)     # B, L/10, 1
                threshold = mean + self.alpha*std
                
                o_list = (o_list.bool() | (torch.abs(inlier_x) > threshold)).float()       # B, L/10, D
                inlier_x = x[:,:,i,:] * (1-o_list)                          # B, L/10, D
            
            s = static_s[:,:,i].unsqueeze(2)                                # 1, L/10, 1
            # quantize the inliers
            if qtype == 'uni':
                inlier_q = F.hardtanh((inlier_x/s).round(), self.qmin, self.qmax)
                inlier_x = inlier_q * s                                     # B, L/10, D
            elif qtype == 'non-uni':
                in_abs = torch.sqrt(inlier_x**2)
                in_sign = inlier_x / (in_abs+ 1e-15)
                in_abs = in_abs / s
                in_abs = torch.clamp(in_abs, 1e-8, 1.0)
                inlier_x = self.log_quantize_efficient(in_abs, in_sign, s)
            else : 
                assert False
            
            # quantize the outliers 
            outlier_x = x[:,:,i,:] * o_list                                 # B, L/10, D
            outlier_s = torch.abs(outlier_x).max(2)[0].clamp(min=1e-8)/o_qmax             # B, L/10
            outlier_s = outlier_s.unsqueeze(2)                              # B, L/10, 1
            outlier_q = F.hardtanh((outlier_x/outlier_s).round(), o_qmin, o_qmax)
            outlier_x = outlier_q * outlier_s
            
            x[:,:,i,:] = inlier_x + outlier_x
        x = x.view(B, new_L, D)                                             # B, L, D
        if concat :
            x = x[:,:L,:]
        return x
  
    def forward(self, x):
        del self.s
        x_copy = x.clone()
        _, _, x_copy, _ = self.extracter_refresh(x_copy, 10)
        b,l,d = x_copy.shape
        x_copy = x_copy.permute(0,2,1)                                      # b, d, l
        x_copy = x_copy.reshape(-1, l)                                      # bd, l
        x_copy = x_copy.permute(1,0)                                        # l, bd
        
        xmax = torch.abs(x_copy).max(1)[0]                                  # l,
        max_val_pos = torch.max(xmax, torch.zeros_like(xmax))
        val = torch.max(max_val_pos, self.eps).unsqueeze(1).unsqueeze(0) / self.qmax
        self.register_parameter("s",torch.nn.Parameter(val))

        x = self.extracter_requant(x, 10, None, self.s, 'uni')
        x = x.half()
        
        return x
    ######################################################################################################
        
class QuantOps(object):
    Act = Q_Act
    Act_h = Q_Act_high
    Linear = Q_Linear
    Linear_h = Q_Linear_high
