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
        x = self.weight #* self.act_func.smooth_scale
        
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
            best_min = xmin.clone()
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
        # print("Q_Linear Max s :" +  str(self.s.max()))
 
    def _weight_quant(self): 
        s = self.s 
        if self.smoothing:
            weight = self.weight #* self.act_func.smooth_scale
        else:
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
        
        # if self.real_int8:
        #     #import pdb; pdb.set_trace()
        #     # print(x.shape)
        #     # print(self.int_weight.shape)
        #     result = vim_GEMM.vim_GEMM(x.contiguous(), \
        #             self.int_weight.contiguous(), \
        #             self.act_func.smooth_scale, \
        #             self.act_func.s, \
        #             self.s, \
        #             16, \
        #             self.qbit)
        #     return result
        if self.n_lv == 0:    
            if self.smoothing:
                weight = self.weight #* self.act_func.smooth_scale
            else:
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
        ####################
        self.n_lva_o = 0
        self.r_sum = 0
        self.r_out = 0
        self.batchsize = 32
        self.q_type = 0
        self.r_block = 0.1
        ####################
        
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
    
    ######################################################################################################
    def extracter_refresh(self, x, n_refresh=10):
        '''
            Extract the outlier channel using list, update new outliers, refresh every n_refresh time steps
            1. get mean, std per batch, per n_step, per time steps -> threshold
            2. outlier extraction
            
            we assume the 3 dimension (B,L,D) for now
        '''
        # reshape input tensor
        ###############################################################
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
        ###############################################################
        # find threshold for each timesteps
        mean = torch.mean(torch.abs(x), dim=3, keepdim=True)            # B, L/10, 10, 1
        std = torch.std(torch.abs(x), dim=3, keepdim=True)              # B, L/10, 10, 1
        threshold = mean + 3*std
        
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

    def extract_outliers(self, x, n_refresh=10, static_o_list=None, static_s=None, qtype='uni'):
        """
        this function detects the outlier list and prepares the inlier and outlier matrices for quantization.
        after this function, x becomes the inlier matrix (with outlier positions zeroed), and the outlier 
        matrix is represented as a sparse coo tensor (with regular precision).

        parameters:
        x (torch.tensor): input tensor of shape [b, l, d], where b is the batch size, l is the sequence 
                            length, and d is the feature dimension.
        n_refresh (int, optional): the number of tokens refreshed per group. l must be divisible by n_refresh.
        static_s (torch.tensor, optional): tensor containing static scaling factors for inlier quantization.
        qtype (str, optional): quantization type; must be "uni" for this function.

        returns:
        static_s (torch.tensor): per-token static scaling factors for inlier quantization (reshaped to [b*l, 1]).
        s_outlier (torch.tensor): per-token scaling factors for outlier quantization (reshaped to [b*l, 1]).
        inlier (torch.tensor): updated inlier matrix (with outlier positions zeroed), flattened to [b*l, d].
        outlier_coo (torch.tensor): sparse coo tensor representing the outlier matrix with shape [b*l, d].
                                    nonzero entries correspond to detected outlier values.
        """
        B, L, D = x.shape
        #print(f"[debug extract_outliers] input shape: b={B}, l={L}, d={D}")
        o_qmax = self.n_lva_o // 2 - 1  # outlier quantization range (max value)
        
        assert L % n_refresh == 0, f"l ({L}) must be divisible by n_refresh ({n_refresh})"
        n_groups = L // n_refresh
        #print(f"[debug extract_outliers] n_groups: {n_groups}")
        
        # reshape input and static scaling factors for grouped processing
        x = x.view(B, n_groups, n_refresh, D)          # [b, n_groups, n_refresh, d]
        static_s = static_s.view(1, n_groups, n_refresh) # [1, n_groups, n_refresh]
        s_outlier_acc = torch.empty(B, n_groups, n_refresh, device=x.device)

        # preallocate a dense outlier tensor (initialize with zeros)
        dense_outlier = torch.zeros_like(x)  # [b, n_groups, n_refresh, d]

        # initialize o_list (mask) for detected outliers
        o_list = torch.zeros(B, n_groups, D, device=x.device)
        
        cnt = 0  # counter for outlier positions
        for i in range(n_refresh):
            # compute inlier component using current mask
            inlier_x = x[:, :, i, :] * (1 - o_list)  # [b, n_groups, d]
            dynamic_s = torch.abs(inlier_x).max(2)[0] / self.qmax  # [b, n_groups]
            
            # update o_list based on threshold computed from inlier_x
            if (dynamic_s > static_s[:, :, i]).any():
                mean_val = torch.mean(torch.abs(inlier_x), dim=2, keepdim=True)
                std_val = torch.std(torch.abs(inlier_x), dim=2, keepdim=True)
                threshold = mean_val + 3 * std_val
                o_list = (o_list.bool() | (torch.abs(inlier_x) > threshold)).float()
                inlier_x = x[:, :, i, :] * (1 - o_list)  # [b, n_groups, d]
            
            # recompute outlier component using updated mask
            dense_outlier[:, :, i, :] = x[:, :, i, :] * o_list
            
            # update inlier portion in x (in-place) with inlier_x
            x[:, :, i, :].copy_(inlier_x)
            
            # compute per-token outlier scaling factors
            s_outlier_acc[:, :, i] = torch.abs(dense_outlier[:, :, i, :]).max(2)[0].clamp(min=1e-8) / o_qmax
            cnt += o_list.sum().float()
        
        rate = cnt / (B * L * D)
        print(f"[debug extract_outliers] total outliers detected: {int(cnt)} out of {B * L * D} elements, rate: {rate}")

        # convert the dense outlier tensor to sparse coo format
        outlier_dense = dense_outlier.view(B * L, D)
        outlier_coo = outlier_dense.to_sparse_coo()
        
        # flatten the updated inlier matrix
        inlier = x.view(B * L, D)
        s_outlier = s_outlier_acc.view(B * L, 1)
        #print(f"[debug extract_outliers] final inlier shape: {inlier.shape}, s_outlier shape: {s_outlier.shape}")
        
        return static_s, s_outlier, inlier, outlier_coo



    def extracter_requant(self, x, n_refresh=10, static_o_list=None, static_s=None, qtype = 'uni'):
        '''
            Extract the outlier channel using list, update new outliers, refresh every n_refresh time steps            
            we assume the 3 dimension (B,L,D) for now
        '''
        # reshape input tensor
        ###############################################################
        B, L, D = x.shape                                              # B, L, D
        #print(f"[DEBUG extracter_requant] Input shape: B={B}, L={L}, D={D}")
        concat = False
        pad = 0
        new_L = L
        o_qmax = self.n_lva_o//2 - 1 # outlier range
        o_qmin = -1*o_qmax
        # DEBUG: notice that the qmin for outlier is negative qmax, but qmin for inlier is 0 
        # so outlier's zero-point is just 0
        
        if L%n_refresh != 0:
            new_L = (L//n_refresh +1) * n_refresh
            pad = new_L - L
            x = torch.cat([x,torch.zeros(B,pad,D,device=x.device)], dim=1) # concat 0s to make L divisible by n_refresh
            static_s = static_s.view(L) 
            # DEBUG: static_s is self.s, passed in from forward. self.s is initially calculated with init
            static_s = torch.cat([static_s,torch.ones(pad,device=static_s.device)], dim=0)    # L -> new_L, concat 1s for static_s
            concat = True
            #print(f"[DEBUG extracter_requant] Padding applied: pad={pad}, new_L={new_L}")


        x = x.view(B, new_L//n_refresh, n_refresh, D)                   # B, L/10, 10, D
        static_s= static_s.view(1, new_L//n_refresh, n_refresh)         # 1, L/10, 10
        o_list = torch.zeros(B, new_L//n_refresh, D, device = x.device) # B, L/10, D # each batch, each sequencem each hidden dimension
        ###############################################################
        cnt=0
        for i in range(n_refresh):
            #print(f"[DEBUG extracter_requant] Iteration {i}")
            # DEBUG: for each sequence
            # exclude outliers in list
            inlier_x = x[:,:,i,:] * (1-o_list)                          # B, L/10, D
            
            # find scale for inliers
            if qtype == 'uni':
                # DEBUG: calculate the max in the D dimension, then quantize by qmax
                # this gives the dynamic scaling factor for inlier
                dynamic_s = torch.abs(inlier_x).max(2)[0]/self.qmax         # B, L/10
            elif qtype == 'non-uni':
                dynamic_s = torch.abs(inlier_x).max(2)[0]                   # B, L/10
            else:
                assert False, "qtype must be either uni or non-uni"
            #print(f"[DEBUG extracter_requant] Iteration {i}: dynamic_s stats: mean={dynamic_s.mean().item():.6f}, max={dynamic_s.max().item():.6f}")
            # new outlier detection, exclusion
            # DEBUG: static_s dimension is B, L/10, 10
            # print(f"dynamic_s shape: {dynamic_s.shape}; static_s[::i] shape: {static_s[:,:,i].shape}")
            if (dynamic_s > static_s[:,:,i]).any(): # all groups
                mean = torch.mean(torch.abs(inlier_x),dim=2,keepdim=True) # B, L/10, 1
                std = torch.std(torch.abs(inlier_x),dim=2,keepdim=True)   # B, L/10, 1
                threshold = mean + 3*std
                #print(f"[DEBUG extracter_requant] Iteration {i}: threshold stats: mean={threshold.mean().item():.6f}, std={threshold.std().item():.6f}")
                # DEBUG: update o_list bitmap, add new elements away from mean with more than 3 std
                o_list = (o_list.bool() | (torch.abs(inlier_x) > threshold)).float()       # B, L/10, D
                #print(f"[DEBUG extracter_requant] Iteration {i}: updated o_list ones count: {o_list.sum().item()}")
                inlier_x = x[:,:,i,:] * (1-o_list)                        # B, L/10, D
            
            s = static_s[:,:,i].unsqueeze(2)                              # 1, L/10, 1
            # quantize the inliers
            if qtype == 'uni':
                # DEBUG: still using the old s to quantize, inlier into scope qmin to qmax
                # DEBUG: what happened to the zero point?
                inlier_q = F.hardtanh((inlier_x/s).round(), self.qmin, self.qmax)
                inlier_x = inlier_q * s                                       # B, L/10, D
            elif qtype == 'non-uni':
                in_abs = torch.sqrt(inlier_x**2)
                in_sign = inlier_x / (in_abs+ 1e-15)
                in_abs = in_abs / s # first divide by s, then clamp to 0,1
                in_abs = torch.clamp(in_abs, 1e-8, 1.0)
                inlier_x = self.log_quantize_efficient(in_abs, in_sign, s) # quantize then dequantize
            else : 
                assert False
            
            # quantize the outliers, this can be stored in as a sparse tensor
            # olist itself can be a sparse tensor, maybe a CSC tensor
            outlier_x = x[:,:,i,:] * o_list                               # B, L/10, D
            # max over D dimension, per-token quantization
            # calculate one scaling factor for outliers in the same dimension, or token
            outlier_s = torch.abs(outlier_x).max(2)[0].clamp(min=1e-8)/o_qmax             # B, L/10
            # print(f"group {i} outlier scale shape: {outlier_s.shape}")
            outlier_s = outlier_s.unsqueeze(2)                            # B, L/10, 1
            outlier_q = F.hardtanh((outlier_x/outlier_s).round(), o_qmin, o_qmax)
            outlier_x = outlier_q * outlier_s # dequantized, outlier always uses uniform quantization
            
            x[:,:,i,:] = inlier_x + outlier_x
            #x[:,:,i,:] = inlier_x # tmp, we ignore the outlier
            group_outlier_cnt = o_list.sum().float()
            cnt += group_outlier_cnt
            #print(f"[DEBUG extracter_requant] Iteration {i}: current cumulative outlier count: {o_list.sum().item()}")
        rate = cnt/(B*new_L*D)
        print(f"[Python, extracter_requant] {int(cnt)} out of {B*new_L*D} outliers detected, rate={rate}")
        x = x.view(B, new_L, D) # B, L, D
        if concat:
            x = x[:,:L,:]
        #print(f"[DEBUG extracter_requant] Final output shape: {x.shape}")
        return x
    
    def initialize(self, n_lv, tensor, n_lva_o, r_sum, r_out, batchsize, q_type, r_block, per_token=False, trunc=False):        
        x = tensor
        # x = tensor / self.smooth_scale # smooth scale is 1.0
        self.n_lv = n_lv
        self.qmax = n_lv - 1
        self.qmin = 0
        self.per_token = per_token     
        ########################
        self.n_lva_o = n_lva_o
        self.r_sum = r_sum
        self.r_out = r_out
        self.batchsize = batchsize
        self.q_type = q_type
        self.r_block = r_block
        ########################
        
        if not trunc:
            if self.per_token:
                b,l,d = x.shape
                x = x.permute(0,2,1) # b, d, l
                x = x.reshape(-1, l) # bd, l
                x = x.permute(1,0) # l, bd
                del self.s
                max_val = x.max(dim=1, keepdim=True)[0] # for each l (token), the largest value
                min_val = x.min(dim=1, keepdim=True)[0] # for each l (token), the smallest value
                val = (max_val - min_val) / self.qmax
                self.register_parameter("s",torch.nn.Parameter(val.unsqueeze(0)))
                # the minimum floating-point value maps to zero in the quantized domain
                # quantize: q = x / s + z, range of q will be [0, qmax]
                # dequantize: x = s * (q - z)
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
            #################################
            for i in range(1, self.num + 1):
                alpha = i / self.num
                tmp_max = xmax * (1-alpha)         
                
                scale = tmp_max
                scale = torch.max(scale, self.eps)
                scale = scale.reshape(new_shape)

                # Perform quantization with the computed scale and zero point
                # sign of x                
                x_abs = torch.sqrt(x**2)
                x_sign = x / (x_abs+ 1e-15)
                x_abs = x_abs / scale
                x_abs = torch.clamp(x_abs, 1e-8, 1.0)
                x_q = self.log_quantize_efficient(x_abs, x_sign, scale)

                # Compute score and update best values
                score = self.lp_loss(x, x_q, 2.4)
                best_max = torch.where(score < best_score, tmp_max, best_max)
                best_score = torch.min(best_score, score)

            # Final scale and zero point calculation
            max_val_pos = torch.max(best_max, torch.zeros_like(best_max))
            del self.s
            val = torch.max(max_val_pos, self.eps).unsqueeze(1).unsqueeze(0)
            self.register_parameter("s",torch.nn.Parameter(val))
            self.z = 0
            #################################
        self.smoothing = True # smoothing is always set to True
        # print("Q_Act Max s :" +  str(self.s.max())) 
        
    def forward(self, x):
        
        if self.real_int8: # Kernel includes act quant procedure
            # TODO cuda kernel perhaps
            return x
        if self.n_lv == 0:
            if self.smoothing:
                return x #/ self.smooth_scale
            else:
                return x
        else:
            ###################################################
            if self.smoothing:
                # the smoothing codepath enables inlier-outlier quantization
                B, L, D = x.shape
                x = self.extracter_requant(x, 10, None, self.s, 'non-uni')
            #####################################################
            else:
                s = self.s
                z = self.z
                x = F.hardtanh(x / s + z, self.qmin, self.qmax) # similar to clamp
                x = RoundQuant.apply(x - z) * s # roundquant is float to int, this line *dequantized* x
            return x
    ######################################################################################################

class Q_Act_u(nn.Module):
    def __init__(self):
        super(Q_Act_u, self).__init__()
        self.n_lv = 0
        # self.n_lv = 64 # fot 6-bit
        self.qmax = self.n_lv // 2 - 1
        self.qmin = -self.qmax
        self.per_channel = False
        self.s = Parameter(torch.Tensor(1))
        self.num = 100
        self.eps = torch.tensor(1e-8)
        # self.smooth_scale = None
        self.smoothing = False
        # self.smoothing = True # for 6-bit
        self.real_int8 = False
        ####################
        self.n_lva_o = 0
        self.r_sum = 0
        self.r_out = 0
        self.batchsize = 32
        self.q_type = 0
        self.r_block = 0.1
        ####################
        
    def quantize_efficient(self, x_round, scale, zero=0):
        q = torch.clamp(x_round + zero, self.qmin , self.qmax)
        return scale * (q - zero)
        
    def log_quantize_efficient(self, x_abs, x_sign, scale):
        # y = clip(round(-log_2(x_abs)))
        # return scale * 2^(-y) * sign(x)
        y = torch.clamp((-1* torch.log2(x_abs)).round(), self.qmin, self.qmax)
        return scale * 2**(-1*y) * x_sign
        
    def log_sqrt2_quantize_efficient(self, x_abs, x_sign, scale):
        # y = clip(round(-log_2(x_abs)))
        # return scale * 2^(-y) * sign(x)
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
    
    ######################################################################################################
    def extracter_refresh(self, x, n_refresh=10):
        '''
            Extract the outlier channel using list, update new outliers, refresh every n_refresh time steps
            1. get mean, std per batch, per n_step, per time steps -> threshold
            2. outlier extraction
            
            we assume the 3 dimension (B,L,D) for now
        '''
        # reshape input tensor
        ###############################################################
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
        ###############################################################
        # find threshold for each timesteps
        mean = torch.mean(torch.abs(x), dim=3, keepdim=True)            # B, L/10, 10, 1
        std = torch.std(torch.abs(x), dim=3, keepdim=True)              # B, L/10, 10, 1
        threshold = mean + 3*std
        
        # find outliers per batch, per n_refresh, per timestep
        outlier_mask = (torch.abs(x) > threshold)                       # B, L/10, 10, D
        # cumulate for each timestep to apply outlier list push
        outlier_mask = outlier_mask.cumsum(dim=2).bool()                # B, L/10, 10, D
        # print("percentage of outliers", outlier_mask.sum().float()/(B*new_L*D))
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
            Extract the outlier channel using list, update new outliers, refresh every n_refresh time steps            
            we assume the 3 dimension (B,L,D) for now
        '''
        # reshape input tensor
        ###############################################################
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
        ###############################################################
        # cnt = 0
        for i in range(n_refresh):
            # exclude outliers in list
            inlier_x = x[:,:,i,:] * (1-o_list)                          # B, L/10, D
            
            # find scale for inliers
            if qtype == 'uni':
                dynamic_s = torch.abs(inlier_x).max(2)[0]/self.qmax         # B, L/10
            elif qtype == 'non-uni':
                dynamic_s = torch.abs(inlier_x).max(2)[0]                   # B, L/10
            else:
                assert False
            
            # new outlier detection, exclusion
            if (dynamic_s > static_s[:,:,i]).any():
                mean = torch.mean(torch.abs(inlier_x),dim=2,keepdim=True) # B, L/10, 1
                std = torch.std(torch.abs(inlier_x),dim=2,keepdim=True)   # B, L/10, 1
                threshold = mean + 3*std
                
                o_list = (o_list.bool() | (torch.abs(inlier_x) > threshold)).float()       # B, L/10, D
                inlier_x = x[:,:,i,:] * (1-o_list)                        # B, L/10, D
            
            s = static_s[:,:,i].unsqueeze(2)                              # 1, L/10, 1
            # quantize the inliers
            if qtype == 'uni':
                inlier_q = F.hardtanh((inlier_x/s).round(), self.qmin, self.qmax)
                inlier_x = inlier_q * s                                       # B, L/10, D
            elif qtype == 'non-uni':
                in_abs = torch.sqrt(inlier_x**2)
                in_sign = inlier_x / (in_abs+ 1e-15)
                in_abs = in_abs / s
                in_abs = torch.clamp(in_abs, 1e-8, 1.0)
                inlier_x = self.log_quantize_efficient(in_abs, in_sign, s)
            else : 
                assert False
            
            # quantize the outliers 
            outlier_x = x[:,:,i,:] * o_list                               # B, L/10, D
            outlier_s = torch.abs(outlier_x).max(2)[0].clamp(min=1e-8)/o_qmax             # B, L/10
            outlier_s = outlier_s.unsqueeze(2)                            # B, L/10, 1
            outlier_q = F.hardtanh((outlier_x/outlier_s).round(), o_qmin, o_qmax)
            outlier_x = outlier_q * outlier_s
            
            x[:,:,i,:] = inlier_x + outlier_x
            # cnt += o_list.sum().float()
        # rate = cnt/(B*new_L*D)
        # if rate > 0.1 : print("         BCD outlier percentage", rate)
        x = x.view(B, new_L, D) # B, L, D
        if concat :
            x = x[:,:L,:]
        return x
    
    def initialize(self, n_lv, tensor, n_lva_o, r_sum, r_out, batchsize, q_type, r_block, per_token=False, trunc=False):        
        # x = tensor / self.smooth_scale # smooth scale is 1.0
        self.n_lv = n_lv
        self.qmax = n_lv//2 - 1
        self.qmin = -1 * self.qmax
        self.per_token = per_token     
        ########################
        self.n_lva_o = n_lva_o
        self.r_sum = r_sum
        self.r_out = r_out
        self.batchsize = batchsize
        self.q_type = q_type
        self.r_block = r_block
        ########################
        
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
            #################################
            for i in range(1, self.num + 1):
                alpha = i / self.num
                tmp_max = xmax * (1-alpha)         
                
                scale = tmp_max / self.qmax
                scale = torch.max(scale, self.eps)
                scale = scale.reshape(new_shape)

                # Perform quantization with the computed scale and zero point
                # sign of x                
                x_q = torch.clamp((x/scale).round(), self.qmin, self.qmax)
                x_q = x_q * scale

                # Compute score and update best values
                score = self.lp_loss(x, x_q, 2.4)
                best_max = torch.where(score < best_score, tmp_max, best_max)
                best_score = torch.min(best_score, score)

            # Final scale and zero point calculation
            max_val_pos = torch.max(best_max, torch.zeros_like(best_max))
            del self.s
            val = torch.max(max_val_pos, self.eps).unsqueeze(1).unsqueeze(0) / self.qmax
            self.register_parameter("s",torch.nn.Parameter(val))
            self.z = 0
            #################################
        self.smoothing = True
        # print("Q_Act Max s :" +  str(self.s.max())) 
        
    def forward(self, x):
        
        if self.real_int8: # Kernel includes act quant procedure
            return x
        if self.n_lv == 0:
            if self.smoothing:
                return x #/ self.smooth_scale
            else:
                return x
        else:
            ###################################################
            if self.smoothing:
                B, L, D = x.shape
                x = self.extracter_requant(x, 10, None, self.s, 'uni')
            #####################################################
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
        # self.n_lv = 64 # fot 6-bit
        self.qmax = self.n_lv // 2 - 1
        self.qmin = -self.qmax
        self.per_channel = False
        self.s = Parameter(torch.Tensor(1))
        self.num = 100
        self.eps = torch.tensor(1e-8)
        # self.smooth_scale = None
        self.smoothing = False
        # self.smoothing = True # for 6-bit
        self.real_int8 = False
        ####################
        self.n_lva_o = 0
        self.r_sum = 0
        self.r_out = 0
        self.batchsize = 32
        self.q_type = 0
        self.r_block = 0.1
        ####################
        
    def quantize_efficient(self, x_round, scale, zero=0):
        q = torch.clamp(x_round + zero, self.qmin , self.qmax)
        return scale * (q - zero)
        
    def log_quantize_efficient(self, x_abs, x_sign, scale):
        # y = clip(round(-log_2(x_abs)))
        # return scale * 2^(-y) * sign(x)
        y = torch.clamp((-1* torch.log2(x_abs)).round(), self.qmin, self.qmax)
        return scale * 2**(-1*y) * x_sign
        
    def log_sqrt2_quantize_efficient(self, x_abs, x_sign, scale):
        # y = clip(round(-log_2(x_abs)))
        # return scale * 2^(-y) * sign(x)
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
    
    ######################################################################################################
    def extracter_refresh(self, x, n_refresh=10):
        '''
            Extract the outlier channel using list, update new outliers, refresh every n_refresh time steps
            1. get mean, std per batch, per n_step, per time steps -> threshold
            2. outlier extraction
            
            we assume the 3 dimension (B,L,D) for now
        '''
        # reshape input tensor
        ###############################################################
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
        ###############################################################
        # find threshold for each timesteps
        mean = torch.mean(torch.abs(x), dim=3, keepdim=True)            # B, L/10, 10, 1
        std = torch.std(torch.abs(x), dim=3, keepdim=True)              # B, L/10, 10, 1
        threshold = mean + 4* std
        
        # find outliers per batch, per n_refresh, per timestep
        outlier_mask = (torch.abs(x) > threshold)                       # B, L/10, 10, D
        # cumulate for each timestep to apply outlier list push
        outlier_mask = outlier_mask.cumsum(dim=2).bool()                # B, L/10, 10, D
        # print("percentage of outliers", outlier_mask.sum().float()/(B*new_L*D))
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

    def extracter_requant(self, x, n_refresh=10, static_o_list=None, static_s=None, static_z =None, qtype = 'uni'):
        '''
            Extract the outlier channel using list, update new outliers, refresh every n_refresh time steps            
            we assume the 3 dimension (B,L,D) for now
        '''
        # reshape input tensor
        ###############################################################
        B, L, D = x.shape                                              # B, L, D
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
            static_s = torch.cat([static_s,torch.ones(pad,device=static_s.device)], dim=0)    # L -> new_L
            static_z = static_z.view(L)
            static_z = torch.cat([static_z,torch.ones(pad,device=static_z.device)], dim=0)    # L -> new_L
            concat = True

        x = x.view(B, new_L//n_refresh, n_refresh, D)                   # B, L/10, 10, D
        static_s= static_s.view(1, new_L//n_refresh, n_refresh)         # 1, L/10, 10
        static_z= static_z.view(1, new_L//n_refresh, n_refresh)         # 1, L/10, 10
        o_list = torch.zeros(B, new_L//n_refresh, D, device = x.device) # B, L/10, D
        ###############################################################
        # cnt = 0
        for i in range(n_refresh):
            # exclude outliers in list
            inlier_x = x[:,:,i,:] * (1-o_list)                          # B, L/10, D
            
            # find scale for inliers
            if qtype == 'uni':
                dynamic_s = (inlier_x.max(2)[0]-inlier_x.min(2)[0])/(self.qmax-self.qmin)       # B, L/10
            elif qtype == 'non-uni':
                dynamic_s = torch.abs(inlier_x).max(2)[0]                   # B, L/10
            else:
                assert False
            
            # new outlier detection, exclusion
            if (dynamic_s > static_s[:,:,i]).any():
                mean = torch.mean(torch.abs(inlier_x),dim=2,keepdim=True) # B, L/10, 1
                std = torch.std(torch.abs(inlier_x),dim=2,keepdim=True)   # B, L/10, 1
                threshold = mean + 4* std
                
                o_list = (o_list.bool() | (torch.abs(inlier_x) > threshold)).float()       # B, L/10, D
                inlier_x = x[:,:,i,:] * (1-o_list)                        # B, L/10, D
            
            s = static_s[:,:,i].unsqueeze(2)                              # 1, L/10, 1
            z = static_z[:,:,i].unsqueeze(2)
            # quantize the inliers
            if False:
                pass
            elif qtype == 'uni':
                # og = inlier_x.clone()
                inlier_q = F.hardtanh((inlier_x/s).round()+z, self.qmin, self.qmax)
                inlier_x = (inlier_q-z) * s                                      # B, L/10, D
                # print("error: ",torch.mean((og-inlier_x).abs())/torch.mean(og.abs()))
            elif qtype == 'non-uni':
                in_abs = torch.sqrt(inlier_x**2)
                in_sign = inlier_x / (in_abs+ 1e-15)
                in_abs = in_abs / s
                in_abs = torch.clamp(in_abs, 1e-8, 1.0)
                inlier_x = self.log_quantize_efficient(in_abs, in_sign, s)
            else : 
                assert False
            
            # quantize the outliers 
            outlier_x = x[:,:,i,:] * o_list                               # B, L/10, D
            outlier_s = (outlier_x.max(2)[0]-outlier_x.min(2)[0]).clamp(min=1e-8)/(o_qmax-o_qmin)             # B, L/10
            outlier_z = torch.round((-1*outlier_x.min(2)[0])/(outlier_s))             # B, L/10
            outlier_s = outlier_s.unsqueeze(2)                            # B, L/10, 1
            outlier_z = outlier_z.unsqueeze(2)                            # B, L/10, 1
            outlier_q = F.hardtanh((outlier_x/outlier_s).round()+outlier_z, o_qmin, o_qmax)
            outlier_x = (outlier_q-outlier_z) * outlier_s
            
            x[:,:,i,:] = inlier_x + outlier_x
            # cnt += o_list.sum().float()
        # rate = cnt/(B*new_L*D)
        # if rate > 0.1 : print("         delta outlier percentage", rate)
        x = x.view(B, new_L, D) # B, L, D
        if concat :
            x = x[:,:L,:]
        return x
    
    def initialize(self, n_lv, tensor, n_lva_o, r_sum, r_out, batchsize, q_type, r_block, per_token=False, trunc=False):        
        # x = tensor / self.smooth_scale # smooth scale is 1.0
        self.n_lv = n_lv
        self.qmax = n_lv - 1
        self.qmin = 0
        self.per_token = per_token     
        ########################
        self.n_lva_o = n_lva_o
        self.r_sum = r_sum
        self.r_out = r_out
        self.batchsize = batchsize
        self.q_type = q_type
        self.r_block = r_block
        ########################
        
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
            
            xmax = x.max(1)[0]   # l,
            xmin = x.min(1)[0]   # l,
            
            new_shape = [-1] + [1] * (len(x.shape) -  1)
            best_score = torch.zeros_like(xmax) + (1e+10)
            best_max = xmax.clone()
            best_min = xmin.clone()
            #################################
            for i in range(1, self.num + 1):
                alpha = i / self.num
                tmp_min = xmin * (1-alpha) + xmax * alpha
                tmp_max = xmin * alpha + xmax * (1-alpha)
                
                scale = torch.max((tmp_max - tmp_min) / (self.qmax - self.qmin), self.eps)
                zero = torch.round(-tmp_min / scale) + self.qmin
                scale = scale.reshape(new_shape)
                zero = zero.reshape(new_shape)

                # Perform quantization with the computed scale and zero point
                # sign of x                
                x_q = torch.clamp((x/scale).round() + zero, self.qmin, self.qmax)
                x_q = (x_q-zero) * scale

                # Compute score and update best values
                score = self.lp_loss(x, x_q, 2.4)
                best_max = torch.where(score < best_score, tmp_max, best_max)
                best_min = torch.where(score < best_score, tmp_min, best_min)
                best_score = torch.min(best_score, score)

            # Final scale and zero point calculation
            max_val_pos = torch.max(best_max, torch.zeros_like(best_max))
            min_val_neg = torch.min(best_min, torch.zeros_like(best_min))
            del self.s
            val = torch.max((max_val_pos - min_val_neg) / (self.qmax - self.qmin), self.eps).unsqueeze(1).unsqueeze(0)
            self.register_parameter("s",torch.nn.Parameter(val))
            self.z = self.qmin - torch.round(min_val_neg.unsqueeze(1).unsqueeze(0) / self.s)
            #################################
        self.smoothing = True
        # print("Q_Act Max s :" +  str(self.s.max())) 
        
    def forward(self, x):
        
        if self.real_int8: # Kernel includes act quant procedure
            return x
        if self.n_lv == 0:
            if self.smoothing:
                return x #/ self.smooth_scale
            else:
                return x
        else:
            ###################################################
            if self.smoothing:
                B, L, D = x.shape
                x = self.extracter_requant(x, 10, None, self.s, self.z, 'uni')
            #####################################################
            else:
                s = self.s
                z = self.z
                x = F.hardtanh(x / s + z, self.qmin, self.qmax)
                x = RoundQuant.apply(x - z) * s
            return x   
    ######################################################################################################


def initialize(layer, input, residual, n_lvw, n_lva, n_lva_o, r_sum, r_out, batchsize, q_type, r_block, act=False, weight=False, per_channel=False, per_token=False, trunc=False):    
    def initialize_hook(module, input, output): 
        if isinstance(module, (Q_Linear)) and weight:
            module.initialize(n_lvw, per_channel=per_channel, trunc=trunc)
        if isinstance(module, (Q_Act)):
            module.initialize(n_lva, input[0], n_lva_o, r_sum, r_out, batchsize, q_type, r_block, per_token=per_token, trunc=trunc)
        if isinstance(module, (Q_Act_u)):
            module.initialize(n_lva, input[0], n_lva_o, r_sum, r_out, batchsize, q_type, r_block, per_token=per_token, trunc=trunc)
        if isinstance(module, (Q_Act_d)):
            module.initialize(n_lva, input[0], n_lva_o, r_sum, r_out, batchsize, q_type, r_block, per_token=per_token, trunc=trunc)

    # print("======init======")
    hooks = []
    for name, module in layer.named_modules():
        hook = module.register_forward_hook(initialize_hook)
        hooks.append(hook)

    with torch.no_grad():
        input = input.to('cuda')
        # residual = residual.to('cuda')
        if isinstance(layer, nn.DataParallel):
            output = layer.module(input, residual)
        else:
            output = layer(input, residual)
            
    for hook in hooks:
        hook.remove()
    # print("======done======")
        
class QuantOps(object):
    initialize = initialize
    Act = Q_Act
    Linear = Q_Linear
    Act_u = Q_Act_u
    Act_d = Q_Act_d
    
