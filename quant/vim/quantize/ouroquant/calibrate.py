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


import copy
import gc
import math
import os
import pdb
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from ouroquant.quantizer import QuantOps as Q
from ouroquant.utils import NativeScalerWithGradNormCount

def get_quant_parameters(model):
    weight_params = []
    act_params = []
    for n, m in model.named_parameters():
        if n.endswith('proj.s'):
            assert m.requires_grad == True
            weight_params.append(m)
        elif n.endswith('act_func.s'):
            assert m.requires_grad == True
            act_params.append(m)

    return iter(weight_params), iter(act_params)

def get_parameters_all(model):
    params = []
    for n, m in model.named_parameters():
        #######################
        if n.endswith('.s'):
        #######################
            params.append(m)
            
    return iter(params)

def set_tunable_parameters(model):
    for n, m in model.named_parameters():
        if n.endswith('.s'):
            m.requires_grad =True
        else:
            m.requires_grad =False

def CALIBRATE(
    model,
    args,
    loader,
    dev,
    static_outlier_list,
):
    print("Starting ...")
    
    for n, m in model.named_parameters():
        m.requires_grad=False
                
    # move embedding layer and first layer to target device
    layers = model.layers    
    layers[0] = layers[0].to(dev)
    dtype = layers[0].mixer.in_proj.weight.dtype
        
    ### hook start for debugging
    outputs = {}
    outputs_res = {}
    inputs = {}
    inputs_res = {}
    dim = model.layers[0].mixer.in_proj.in_features
    seq = 197 
    inps = torch.zeros(
        (args.batch_size, seq, dim), dtype=dtype, device=dev
    )
    
    num_epoch = 0
    def hook_fn(module, input, output):
        if 'layers.0' in module.name : #or 'layers.1' in module.name or 'layers.23' in module.name:
            if module.name not in inputs:
                inputs[module.name] = torch.zeros((args.batch_size, seq, dim), dtype=dtype, device=dev)
                inputs_res[module.name] = torch.zeros((args.batch_size, seq, dim), dtype=dtype, device=dev) 
            if module.name not in outputs:
                outputs[module.name] = torch.zeros((args.batch_size, seq, output[0].shape[-1]), dtype=dtype, device=dev)
                outputs_res[module.name] = torch.zeros((args.batch_size, seq, dim), dtype=dtype, device=dev)            

            begin = (num_epoch) * int(args.train_batch)
            end = (num_epoch + 1) * int(args.train_batch)       
            
            inputs[module.name][begin:end] = input[0][:]
            if input[1] is not None:
                inputs_res[module.name][begin:end] = input[1][:]
            else:
                inputs_res[module.name] = None
            outputs[module.name][begin:end] = output[0][:]
            outputs_res[module.name][begin:end] = output[1][:] 
                
    hooks = []

    from tools.models_mamba import Block
    for name, module in model.named_modules():
        if isinstance(module, Block):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
            

    model.eval()

    for i, (input, target) in enumerate(loader):
        # print("Batch labels:", target.tolist())
        input = input.to(dev)
        with torch.no_grad():   
            for i in range(args.batch_size//int(args.train_batch)):
                index = i * int(args.train_batch)
                out = model(input[index:index+int(args.train_batch)])
                num_epoch += 1
        break
    
            
    for hook in hooks:
        hook.remove()

    layers[0] = layers[0].cpu()

    fp_inps = inputs['layers.0']
    quant_inps = inputs['layers.0'].clone()
    fp_residual = fp_inps.clone()
    quant_residual = copy.deepcopy(inps)
    
    loss_func = torch.nn.MSELoss()

    layer_name_prefix = "layers"
    pairs = {
        "in_proj"  :"in"    ,
        "out_proj" :"out"   ,
        "x_proj"   :"x_p"   ,
        "x_proj_b" :"x_pb"  ,
        "dt_proj"  :"dt_p"  ,
        "dt_proj_b":"dt_pb" ,
    }
    
    omni_parameters = {}
    for i in range(len(layers)):
        print(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        qlayer = copy.deepcopy(layer)   
        qlayer = qlayer.to(dev)
        fp_inps_0 = fp_inps.clone()
        fp_residual_0 = None if i==0 else fp_residual.clone()
        if args.epochs > 0:
            with torch.no_grad():
                for j in range(args.batch_size):
                    if i==0:
                        fp_inps[j], fp_residual[j] = qlayer(fp_inps[j].unsqueeze(0), None)
                    else:
                        fp_inps[j], fp_residual[j] = qlayer(fp_inps[j].unsqueeze(0), fp_residual[j].unsqueeze(0))
        
        # for name, module in qlayer.named_modules():
        #     if isinstance(module, Q.Linear):
        #         if module.act_func is not None:
        #             for key in pairs.keys():
        #                 if key in name:
        #                     list = static_outlier_list[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=module.weight.dtype)
        #                     del module.act_func.static_o_list
        #                     module.act_func.register_parameter("static_o_list",torch.nn.Parameter(list))

        Q.initialize(qlayer, fp_inps_0, fp_residual_0, args.n_lvw, args.n_lva, args.n_lva_o, act=False, weight=True, per_channel=True, per_token=True, trunc=True)
        
        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()
                quant_inps = quant_inps.float()
                quant_residual = quant_residual.float()
            set_tunable_parameters(qlayer)
            
            weight_params, act_params = get_quant_parameters(qlayer)
            optimizer = torch.optim.AdamW(
                [{"params":weight_params,"lr":args.lr_w}, 
                 {"params":act_params, "lr":args.lr_a}
                 ],weight_decay=1e-5)
            loss_scaler = NativeScalerWithGradNormCount()

            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.batch_size//int(args.train_batch)):    
                    index = j * int(args.train_batch)
                    if i==0:
                        quant_out, _ = qlayer(quant_inps[index:index+int(args.train_batch),], None)
                    else:                              
                        quant_out, _ = qlayer(quant_inps[index:index+int(args.train_batch),], quant_residual[index:index+int(args.train_batch),])
                    fp = fp_inps[index:index+int(args.train_batch),].float()
                    qout = quant_out.float()                        
                    loss = (1 - F.cosine_similarity(fp, qout, dim=-1, eps=1e-8)).mean()
                    if not math.isfinite(loss.item()):
                        print("Loss is NAN, stopping training")
                        breakpoint()

                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,clip_grad=2.0, parameters= get_parameters_all(qlayer)).cpu()
                    norm_list.append(norm.data)
                
                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                if epochs % 50 == 0 or epochs == args.epochs-1:
                    print(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2} ")
            del optimizer
        if args.epochs>0:
            with torch.no_grad():
                for j in range(args.batch_size//int(args.train_batch)):    
                    index = j * int(args.train_batch)
                    if i==0:
                        quant_inps[index:index+int(args.train_batch),], quant_residual[index:index+int(args.train_batch),] = qlayer(quant_inps[index:index+int(args.train_batch),], None)
                    else:                              
                        quant_inps[index:index+int(args.train_batch),], quant_residual[index:index+int(args.train_batch),] = qlayer(quant_inps[index:index+int(args.train_batch),], quant_residual[index:index+int(args.train_batch),])               
                
            layers[i] = qlayer.to("cpu")
        else:
            layers[i] = qlayer.to("cpu")
      
        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps

    torch.cuda.empty_cache()
    gc.collect()
    return model.to(dev)