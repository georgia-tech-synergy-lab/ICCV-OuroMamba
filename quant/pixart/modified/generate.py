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
# Modified from PixArt-sigma repos: https://github.com/PixArt-alpha/PixArt-sigma


import torch
from diffusers import Transformer2DModel, PixArtSigmaPipeline

from ouroquant.quantizer import QuantOps as Q
import torch.nn as nn
import argparse

def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)  
# args passed
def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16

    transformer = Transformer2DModel.from_pretrained(
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", 
        subfolder='transformer', 
        torch_dtype=weight_dtype,
        use_safetensors=True,
        cache_dir="/tmp"
    )
    pipe = PixArtSigmaPipeline.from_pretrained(
        "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
        transformer=transformer,
        torch_dtype=weight_dtype,
        use_safetensors=True,
        cache_dir="/tmp"
    )
    pipe.to(device)

    torch.manual_seed(42) 

    ###############################################################
    if args.mode == 'w4a8':
        activation_func = Q.Act_h()
    elif args.mode == 'w4a4':
        activation_func = Q.Act()
    else:
        activation_func = None
    ###############################################################

    for name, module in transformer.named_modules():
        if isinstance(module, nn.Linear) and args.mode != 'fp16':
            if 'to_q' in name  or 'to_k' in name  or 'to_v' in name or 'to_out.0' in name: 
                quantlinear = Q.Linear(module.in_features, module.out_features, 
                                act_func=activation_func,
                                bias=False if module.bias is None else True,
                                device=module.weight.device)
                quantlinear.weight.data = module.weight
                if module.bias is not None:
                    quantlinear.bias.data = module.bias
                add_new_module(name, transformer, quantlinear)
                del quantlinear
            if 'net.0.proj' in name or 'net.2' in name:
                quantlinear = Q.Linear_h(module.in_features, module.out_features, 
                                act_func=activation_func,
                                bias=False if module.bias is None else True,
                                device=module.weight.device)
                quantlinear.weight.data = module.weight
                if module.bias is not None:
                    quantlinear.bias.data = module.bias
                add_new_module(name, transformer, quantlinear)
                del quantlinear

    for name, module in transformer.named_modules():
        module.name = name 
            

    dir = "./images"
    prompt = args.prompt
    image = pipe(prompt).images[0]
    image.save(f"{dir}/{args.filename}")
    print(f"Image saved to {dir}/{args.filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PixArt Sigma Quantization")
    # parser.add_argument('--w4a8', action='store_true', help='Use W4A8 quantization')
    parser.add_argument('--mode', type=str, choices=['w4a8', 'w4a4', 'fp16'], default='fp16', help='Quantization mode: w4a8, w8a8, or fp16')
    parser.add_argument('--prompt', type=str,
                        default= "A realistic image of a penguin wearing a hat and riding a bicycle on a crowded beach. Colorful beach balls and seagulls surround it.",
                        #"Hyper-realistic, ultra-detailed depiction of several brightly colored rocks scattered on a vibrant, multicolored beach. The rocks appear like luminous spheres, glowing with radiant hues, reflecting light with glass-like clarity. The sand and ocean shimmer with saturated tones, creating a surreal yet photorealistic atmosphere. Rendered in ultra high resolution, cinematic lighting, volumetric glow, and crisp fine textures.",
                        help='Prompt for image generation')
    parser.add_argument('--filename', type=str, default="output_image.png", help='Filename for the output image')
    args = parser.parse_args()

    main(args)
    print("PixArt Sigma quantization completed successfully.")
