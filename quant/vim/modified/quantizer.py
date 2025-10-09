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
        self.n_lv = 0
        self.qmax = self.n_lv // 2 - 1 
        self.qmin = -self.qmax
        self.per_channel = False
        self.s = Parameter(torch.Tensor(1))
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
            self.qbit = 4

    def initialize(self, n_lv, per_channel=False, trunc=False):
        x = self.weight
        self.n_lv = n_lv
        self.qmax = n_lv // 2 - 1
        self.qmin = -self.qmax
        self.per_channel = per_channel     
        if not trunc:
            if self.per_channel:
                del self.s
                max_val = x.abs().max(dim=1, keepdim=True)[0]
                val = max_val / self.qmax
                self.register_parameter("s",torch.nn.Parameter(val))
                
            else:
                max_val = x.abs().max()
                self.s.data = max_val / self.qmax
        else:
            if self.per_channel:
                x = x.flatten(1)
            else:
                x = x.flatten().unsqueeze(0)

            xmin = x.min(1)[0]
            xmax = x.max(1)[0]
            
            if self.per_channel:
                new_shape = [-1] + [1] * (len(x.shape) -  1)

            best_score = torch.zeros_like(xmin) + (1e+10)
            best_max = xmax.clone()
            xrange = torch.max(xmin.abs(), xmax)
            
            for i in range(1, self.num + 1):
                tmp_max = xrange / self.num * i
                scale = torch.max(tmp_max / self.qmax, self.eps)
                if self.per_channel:
                    scale = scale.reshape(new_shape)
                x_round = torch.round(x/scale)
                x_q = self.quantize_efficient(x_round, scale)
                score = self.lp_loss(x, x_q, 2.4)
                best_max = torch.where(score < best_score, tmp_max, best_max)
                best_score = torch.min(score, best_score)                
            max_val = torch.max(best_max, torch.zeros_like(best_max))
            
            if self.per_channel:
                del self.s
                val = torch.max(max_val / self.qmax, self.eps).unsqueeze(1)
                self.register_parameter("s",torch.nn.Parameter(val))
            else:
                self.s.data = torch.max(max_val / self.qmax, self.eps)

        self.smoothing = True
 
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

        if self.n_lv == 0:    
            weight = self.weight
            return F.linear(x, weight, self.bias)
        else:
            try:
                weight = self._weight_quant()
            except:
                breakpoint()
            return F.linear(x, weight, self.bias)

class Q_Act(nn.Module):
    def __init__(self):
        super(Q_Act, self).__init__()
        self.n_lv = 0
        self.qmax = self.n_lv // 2 - 1
        self.qmin = -self.qmax
        self.per_channel = False
        self.s = Parameter(torch.Tensor(1))
        self.num = 100
        self.eps = torch.tensor(1e-8)
        self.smoothing = False
        self.real_int8 = False
        self.n_lva_o = 0
        
    def quantize_efficient(self, x_round, scale, zero=0):
        q = torch.clamp(x_round + zero, self.qmin , self.qmax)
        return scale * (q - zero)
        
    def log_quantize_efficient(self, x_abs, x_sign, scale):
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
        # reshape input tensor per n_refresh timesteps
        B, L, D = x.shape                                              # B, L, D
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
        threshold = mean + 3*std
        
        # find outliers per batch, per n_refresh, per timestep
        outlier_mask = (torch.abs(x) > threshold)                       # B, L/10, 10, D
        
        # accumulate for each refresh groups to apply outlier list
        outlier_mask = outlier_mask.cumsum(dim=2).bool()                # B, L/10, 10, D
        inlier_mask = ~outlier_mask
        
        # extract the out/inliers
        outlier_x = torch.where(outlier_mask, x, torch.zeros_like(x))   # (B, L/10, 10, D)
        inlier_x = torch.where(inlier_mask, x, torch.zeros_like(x))     # (B, L/10, 10, D)
        
        # reshape to original shape
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
        B, L, D = x.shape                                              # B, L, D
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
        x = x.view(B, new_L//n_refresh, n_refresh, D)                   # B, L/10, 10, D
        
        static_s= static_s.view(1, new_L//n_refresh, n_refresh)         # 1, L/10, 10
        o_list = torch.zeros(B, new_L//n_refresh, D, device = x.device) # B, L/10, D
        if static_o_list is not None:
            o_list = o_list.bool() | static_o_list.bool().unsqueeze(0).unsqueeze(0) 
            o_list= o_list.float()

        # mixed quantization within n_refresh time steps per timestep
        for i in range(n_refresh):
            # calculate threshold for new outlier detection
            inlier_x = x[:,:,i,:] * (1-o_list)                          # B, L/10, D
            if qtype == 'uni':
                dynamic_s = torch.abs(inlier_x).max(2)[0]/self.qmax     # B, L/10
            elif qtype == 'non-uni':
                dynamic_s = torch.abs(inlier_x).max(2)[0]               # B, L/10
            else:
                assert False
                
            # new outlier detection, exclusion
            if (dynamic_s > static_s[:,:,i]).any():
                mean = torch.mean(torch.abs(inlier_x),dim=2,keepdim=True) # B, L/10, 1
                std = torch.std(torch.abs(inlier_x),dim=2,keepdim=True)   # B, L/10, 1
                threshold = mean + 3*std
                o_list = (o_list.bool() | (torch.abs(inlier_x) > threshold)).float()
                inlier_x = x[:,:,i,:] * (1-o_list)                        # B, L/10, D
            
            # inlier quantization (static symetric quantization)
            s = static_s[:,:,i].unsqueeze(2)                              # 1, L/10, 1
            if qtype == 'uni':
                inlier_q = F.hardtanh((inlier_x/s).round(), self.qmin, self.qmax)
                inlier_x = inlier_q * s                                   # B, L/10, D
            elif qtype == 'non-uni':
                in_abs = torch.sqrt(inlier_x**2)
                in_sign = inlier_x / (in_abs+ 1e-15)
                in_abs = in_abs / s
                in_abs = torch.clamp(in_abs, 1e-8, 1.0)
                inlier_x = self.log_quantize_efficient(in_abs, in_sign, s)
            else : 
                assert False
            
            # outlier quantization (dynamic symetric quantization)
            if self.n_lva_o == 2**16:
                outlier_x = x[:,:,i,:] * o_list                               # B, L/10, D
            else:
                outlier_x = x[:,:,i,:] * o_list                               # B, L/10, D
                outlier_s = torch.abs(outlier_x).max(2)[0].clamp(min=1e-8)/o_qmax
                outlier_s = outlier_s.unsqueeze(2)                            # B, L/10, 1
                outlier_q = F.hardtanh((outlier_x/outlier_s).round(), o_qmin, o_qmax)
                outlier_x = outlier_q * outlier_s
            
            x[:,:,i,:] = inlier_x + outlier_x

        # reshape to original shape
        x = x.view(B, new_L, D)                                           # B, L, D
        if concat :
            x = x[:,:L,:]
        return x
    
    def initialize(self, n_lv, tensor, n_lva_o, per_token=False, trunc=False):
        self.n_lv = n_lv
        self.qmax = n_lv//2 - 1
        self.qmin = -1 * self.qmax
        self.per_token = per_token     
        self.n_lva_o = n_lva_o
        
        if not trunc:
            if self.per_token:
                b,l,d = x.shape
                x = x.permute(0,2,1) # b, d, l
                x = x.reshape(-1, l) # bd, l
                x = x.permute(1,0) # l, bd
                del self.s
                max_val = x.max(dim=1, keepdim=True)[0]
                min_val = x.min(dim=1, keepdim=True)[0]
                val = (max_val - min_val) / self.qmax
                self.register_parameter("s",torch.nn.Parameter(val.unsqueeze(0)))
                self.z = torch.round(-min_val.unsqueeze(0) / self.s)
                
            else:
                max_val = x.max()
                min_val = x.min()
                val = (max_val - min_val) / self.qmax
                self.s.data = torch.tensor(val)
                self.z = torch.round(-min_val / self.s)
        else:
            _, _, x, _ = self.extracter_refresh(tensor, 10)
            
            b,l,d = x.shape
            x = x.permute(0,2,1) # b, d, l
            x = x.reshape(-1, l) # bd, l
            x = x.permute(1,0)   # l, bd
            
            xmax = torch.abs(x).max(1)[0] # l,
            
            new_shape = [-1] + [1] * (len(x.shape) -  1)
            best_score = torch.zeros_like(xmax) + (1e+10)
            best_max = xmax.clone()
            for i in range(1, self.num + 1):
                alpha = i / self.num
                tmp_max = xmax * (1-alpha)         
                
                scale = tmp_max / self.qmax
                scale = torch.max(scale, self.eps)
                scale = scale.reshape(new_shape)

                x_q = torch.clamp((x/scale).round(), self.qmin, self.qmax)
                x_q = x_q * scale

                score = self.lp_loss(x, x_q, 2.4)
                best_max = torch.where(score < best_score, tmp_max, best_max)
                best_score = torch.min(best_score, score)

            max_val_pos = torch.max(best_max, torch.zeros_like(best_max))
            del self.s
            val = torch.max(max_val_pos, self.eps).unsqueeze(1).unsqueeze(0) / self.qmax
            self.register_parameter("s",torch.nn.Parameter(val))
            self.z = 0
        self.smoothing = True
        
    def forward(self, x):
        if self.n_lv == 0:
            return x
        else:
            if self.smoothing:
                B, L, D = x.shape
                x = self.extracter_requant(x, 10, None, self.s, 'uni')
                # x = self.extracter_requant(x, 10, self.static_o_list, self.s, 'uni')
            else:
                s = self.s
                z = self.z
                x = F.hardtanh(x / s + z, self.qmin, self.qmax)
                x = RoundQuant.apply(x - z) * s
            return x
    ######################################################################################################

class Q_Act_d(nn.Module):
    def __init__(self):
        super(Q_Act_d, self).__init__()
        self.n_lv = 0
        self.qmax = self.n_lv // 2 - 1
        self.qmin = -self.qmax
        self.per_channel = False
        self.s = Parameter(torch.Tensor(1))
        self.num = 100
        self.eps = torch.tensor(1e-8)
        self.smoothing = False
        self.real_int8 = False
        self.n_lva_o = 0
        
    def quantize_efficient(self, x_round, scale, zero=0):
        q = torch.clamp(x_round + zero, self.qmin , self.qmax)
        return scale * (q - zero)
        
    def log_quantize_efficient(self, x_abs, x_sign, scale):
        y = torch.clamp((-1* torch.log2(x_abs)).round(), self.qmin, self.qmax)
        return scale * 2**(-1*y) * x_sign
        
    def log_sqrt2_quantize_efficient(self, x_abs, x_sign, scale):
        x_abs = -1*torch.log2(x_abs)*2
        x_abs = x_abs.round()
        y = torch.clamp(x_abs, self.qmin, self.qmax * 2)
        odd_mask = (x_abs % 2) * (math.sqrt(2) - 1) + 1
        return 2**(-1*torch.ceil(y/2)) * odd_mask * scale * x_sign
    
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
        B, L, D = x.shape
        concat = False
        pad = 0
        new_L = L
        if L%n_refresh != 0:
            new_L = (L//n_refresh +1) * n_refresh
            pad = new_L - L
            x = torch.cat([x,torch.zeros(B,pad,D, device=x.device)], dim=1)
            concat = True

        x = x.view(B, new_L//n_refresh, n_refresh, D)
        mean = torch.mean(torch.abs(x), dim=3, keepdim=True) 
        std = torch.std(torch.abs(x), dim=3, keepdim=True) 
        threshold = mean + 4* std
        
        outlier_mask = (torch.abs(x) > threshold) 
        outlier_mask = outlier_mask.cumsum(dim=2).bool()
        inlier_mask = ~outlier_mask
        
        outlier_x = torch.where(outlier_mask, x, torch.zeros_like(x))
        inlier_x = torch.where(inlier_mask, x, torch.zeros_like(x))  
        
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

    def extracter_requant(self, x, n_refresh=10, static_o_list=None, static_s=None, static_z =None, qtype = 'uni'):
        B, L, D = x.shape
        concat = False
        pad = 0
        new_L = L
        o_qmax = self.n_lva_o - 1
        o_qmin = 0
        
        if L%n_refresh != 0:
            new_L = (L//n_refresh +1) * n_refresh
            pad = new_L - L
            x = torch.cat([x,torch.zeros(B,pad,D,device=x.device)], dim=1)
            static_s = static_s.view(L)
            static_s = torch.cat([static_s,torch.ones(pad,device=static_s.device)], dim=0)
            static_z = static_z.view(L)
            static_z = torch.cat([static_z,torch.zeros(pad,device=static_z.device)], dim=0)
            concat = True

        x = x.view(B, new_L//n_refresh, n_refresh, D)                   
        static_s= static_s.view(1, new_L//n_refresh, n_refresh)         
        static_z= static_z.view(1, new_L//n_refresh, n_refresh)         
        o_list = torch.zeros(B, new_L//n_refresh, D, device = x.device) 
        for i in range(n_refresh):
            inlier_x = x[:,:,i,:] * (1-o_list)
            if qtype == 'uni':
                dynamic_s = (inlier_x.max(2)[0]-inlier_x.min(2)[0])/(self.qmax-self.qmin)
            elif qtype == 'non-uni':
                dynamic_s = torch.abs(inlier_x).max(2)[0]
            else:
                assert False
            if (dynamic_s > static_s[:,:,i]).any():
                mean = torch.mean(torch.abs(inlier_x),dim=2,keepdim=True)
                std = torch.std(torch.abs(inlier_x),dim=2,keepdim=True)  
                threshold = mean + 4* std
                
                o_list = (o_list.bool() | (torch.abs(inlier_x) > threshold)).float()
                inlier_x = x[:,:,i,:] * (1-o_list)
            
            s = static_s[:,:,i].unsqueeze(2)      
            z = static_z[:,:,i].unsqueeze(2)
            if False:
                pass
            elif qtype == 'uni':
                inlier_q = F.hardtanh((inlier_x/s).round()+z, self.qmin, self.qmax)
                inlier_x = (inlier_q-z) * s
            elif qtype == 'non-uni':
                in_abs = torch.sqrt(inlier_x**2)
                in_sign = inlier_x / (in_abs+ 1e-15)
                in_abs = in_abs / s
                in_abs = torch.clamp(in_abs, 1e-8, 1.0)
                inlier_x = self.log_quantize_efficient(in_abs, in_sign, s)
            else : 
                assert False
            
            if self.n_lva_o == 2**16:
                outlier_x = x[:,:,i,:] * o_list                           
            else:
                outlier_x = x[:,:,i,:] * o_list
                outlier_s = (outlier_x.max(2)[0]-outlier_x.min(2)[0]).clamp(min=1e-8)/(o_qmax-o_qmin)
                outlier_z = torch.round((-1*outlier_x.min(2)[0])/(outlier_s))
                outlier_s = outlier_s.unsqueeze(2)
                outlier_z = outlier_z.unsqueeze(2)
                outlier_q = F.hardtanh((outlier_x/outlier_s).round()+outlier_z, o_qmin, o_qmax)
                outlier_x = (outlier_q-outlier_z) * outlier_s
            
            x[:,:,i,:] = inlier_x + outlier_x
        x = x.view(B, new_L, D)
        if concat :
            x = x[:,:L,:]
        return x
    
    def initialize(self, n_lv, tensor, n_lva_o, per_token=False, trunc=False):
        self.n_lv = n_lv
        self.qmax = n_lv - 1
        self.qmin = 0
        self.per_token = per_token     
        self.n_lva_o = n_lva_o
        
        if not trunc:
            if self.per_token:
                b,l,d = x.shape
                x = x.permute(0,2,1) 
                x = x.reshape(-1, l) 
                x = x.permute(1,0)   
                del self.s
                max_val = x.max(dim=1, keepdim=True)[0]
                min_val = x.min(dim=1, keepdim=True)[0]
                val = (max_val - min_val) / self.qmax
                self.register_parameter("s",torch.nn.Parameter(val.unsqueeze(0)))
                self.z = torch.round(-min_val.unsqueeze(0) / self.s)
                
            else:
                max_val = x.max()
                min_val = x.min()
                val = (max_val - min_val) / self.qmax
                self.s.data = torch.tensor(val)
                self.z = torch.round(-min_val / self.s)
        else:
            _, _, x, _ = self.extracter_refresh(tensor, 10)
            
            b,l,d = x.shape
            x = x.permute(0,2,1) 
            x = x.reshape(-1, l) 
            x = x.permute(1,0)   
            
            xmax = x.max(1)[0]   
            xmin = x.min(1)[0]   
            
            new_shape = [-1] + [1] * (len(x.shape) -  1)
            best_score = torch.zeros_like(xmax) + (1e+10)
            best_max = xmax.clone()
            best_min = xmin.clone()
            for i in range(1, self.num + 1):
                alpha = i / self.num
                tmp_min = xmin * (1-alpha) + xmax * alpha
                tmp_max = xmin * alpha + xmax * (1-alpha)
                
                scale = torch.max((tmp_max - tmp_min) / (self.qmax - self.qmin), self.eps)
                zero = torch.round(-tmp_min / scale) + self.qmin
                scale = scale.reshape(new_shape)
                zero = zero.reshape(new_shape)

                x_q = torch.clamp((x/scale).round() + zero, self.qmin, self.qmax)
                x_q = (x_q-zero) * scale

                score = self.lp_loss(x, x_q, 2.4)
                best_max = torch.where(score < best_score, tmp_max, best_max)
                best_min = torch.where(score < best_score, tmp_min, best_min)
                best_score = torch.min(best_score, score)

            max_val_pos = torch.max(best_max, torch.zeros_like(best_max))
            min_val_neg = torch.min(best_min, torch.zeros_like(best_min))
            del self.s
            val = torch.max((max_val_pos - min_val_neg) / (self.qmax - self.qmin), self.eps).unsqueeze(1).unsqueeze(0)
            self.register_parameter("s",torch.nn.Parameter(val))
            self.z = self.qmin - torch.round(min_val_neg.unsqueeze(1).unsqueeze(0) / self.s)
        self.smoothing = True
        
    def forward(self, x):
        if self.real_int8:
            return x
        if self.n_lv == 0:
            return x
        else:
            if self.smoothing:
                B, L, D = x.shape
                x = self.extracter_requant(x, 10, None, self.s, self.z, 'uni')
            else:
                s = self.s
                z = self.z
                x = F.hardtanh(x / s + z, self.qmin, self.qmax)
                x = RoundQuant.apply(x - z) * s
            return x   


def initialize(layer, input, residual, n_lvw, n_lva, n_lva_o, act=False, weight=False, per_channel=False, per_token=False, trunc=False):    
    def initialize_hook(module, input, output): 
        if isinstance(module, (Q_Linear)) and weight:
            module.initialize(n_lvw, per_channel=per_channel, trunc=trunc)
        if isinstance(module, (Q_Act)):
            module.initialize(n_lva, input[0], n_lva_o, per_token=per_token, trunc=trunc)
        if isinstance(module, (Q_Act_d)):
            module.initialize(n_lva, input[0], n_lva_o, per_token=per_token, trunc=trunc)

    hooks = []
    for name, module in layer.named_modules():
        hook = module.register_forward_hook(initialize_hook)
        hooks.append(hook)

    with torch.no_grad():
        input = input.to('cuda')
        if isinstance(layer, nn.DataParallel):
            output = layer.module(input, residual)
        else:
            output = layer(input, residual)
            
    for hook in hooks:
        hook.remove()
        
class QuantOps(object):
    initialize = initialize
    Act = Q_Act
    Linear = Q_Linear
    Act_d = Q_Act_d
    
