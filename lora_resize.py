#
# File from: https://raw.githubusercontent.com/mgz-dev/sd-scripts/main/networks/resize_lora.py
#

# Convert LoRA to different rank approximation (should only be used to go to lower rank)
# This code is based off the extract_lora_from_models.py file which is based on https://github.com/cloneofsimo/lora/blob/develop/lora_diffusion/cli_svd.py
# Thanks to cloneofsimo and kohya

import torch
from tqdm import tqdm
import numpy as np

from .utility import index_sv_cumulative, index_sv_fro

MIN_SV = 1e-6


def resize_lora_model(lora_sd, new_rank, save_dtype, device, dynamic_method, dynamic_param, verbose):
    network_alpha = None
    network_dim = None
    verbose_str = "\n"
    fro_list = []
    save_dtype = str_to_dtype(save_dtype)

    # Extract loaded lora dim and alpha
    for key, value in lora_sd.items():
        if network_alpha is None and 'alpha' in key:
            network_alpha = value
        if network_dim is None and 'lora_down' in key and len(value.size()) == 2:
            network_dim = value.size()[0]
        if network_alpha is not None and network_dim is not None:
            break
        if network_alpha is None:
            network_alpha = network_dim

    scale = network_alpha / network_dim

    if dynamic_method:
        print(
            f"Dynamically determining new alphas and dims based off {dynamic_method}: {dynamic_param}, max rank is {new_rank}")

    lora_down_weight = None
    lora_up_weight = None

    o_lora_sd = lora_sd.copy()
    block_down_name = None
    block_up_name = None

    with torch.no_grad():
        for key, value in tqdm(lora_sd.items()):
            if 'lora_down' in key:
                block_down_name = key.split(".")[0]
                lora_down_weight = value
            if 'lora_up' in key:
                block_up_name = key.split(".")[0]
                lora_up_weight = value

            weights_loaded = (lora_down_weight is not None and lora_up_weight is not None)

            if (block_down_name == block_up_name) and weights_loaded:

                conv2d = (len(lora_down_weight.size()) == 4)

                if conv2d:
                    full_weight_matrix = merge_conv(lora_down_weight, lora_up_weight, device)
                    param_dict = extract_conv(full_weight_matrix, new_rank, dynamic_method, dynamic_param, device,
                                              scale)
                else:
                    full_weight_matrix = merge_linear(lora_down_weight, lora_up_weight, device)
                    param_dict = extract_linear(full_weight_matrix, new_rank, dynamic_method, dynamic_param, device,
                                                scale)

                if verbose:
                    max_ratio = param_dict['max_ratio']
                    sum_retained = param_dict['sum_retained']
                    fro_retained = param_dict['fro_retained']
                    if not np.isnan(fro_retained):
                        fro_list.append(float(fro_retained))

                    verbose_str += f"{block_down_name:75} | "
                    verbose_str += f"sum(S) retained: {sum_retained:.1%}, fro retained: {fro_retained:.1%}, max(S) ratio: {max_ratio:0.1f}"

                if verbose and dynamic_method:
                    verbose_str += f", dynamic | dim: {param_dict['new_rank']}, alpha: {param_dict['new_alpha']}\n"
                else:
                    verbose_str += f"\n"

                new_alpha = param_dict['new_alpha']
                o_lora_sd[block_down_name + "." + "lora_down.weight"] = param_dict["lora_down"].to(
                    save_dtype).contiguous()
                o_lora_sd[block_up_name + "." + "lora_up.weight"] = param_dict["lora_up"].to(save_dtype).contiguous()
                o_lora_sd[block_up_name + "." "alpha"] = torch.tensor(param_dict['new_alpha']).to(save_dtype)

                block_down_name = None
                block_up_name = None
                lora_down_weight = None
                lora_up_weight = None
                weights_loaded = False
                del param_dict

    if verbose:
        print(verbose_str)

        print(f"Average Frobenius norm retention: {np.mean(fro_list):.2%} | std: {np.std(fro_list):0.3f}")
    print("resizing complete")
    return o_lora_sd, network_dim, new_alpha


# Modified from Kohaku-blueleaf's extract/merge functions
def extract_conv(weight, lora_rank, dynamic_method, dynamic_param, device, scale=1):
    out_size, in_size, kernel_size, _ = weight.size()
    U, S, Vh = torch.linalg.svd(weight.reshape(out_size, -1).to(device))

    param_dict = rank_resize(S, lora_rank, dynamic_method, dynamic_param, scale)
    lora_rank = param_dict["new_rank"]

    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S)
    Vh = Vh[:lora_rank, :]

    param_dict["lora_down"] = Vh.reshape(lora_rank, in_size, kernel_size, kernel_size).cpu()
    param_dict["lora_up"] = U.reshape(out_size, lora_rank, 1, 1).cpu()
    del U, S, Vh, weight
    return param_dict


def rank_resize(S, rank, dynamic_method, dynamic_param, scale=1):
    param_dict = {}

    if dynamic_method == "sv_ratio":
        # Calculate new dim and alpha based off ratio
        max_sv = S[0]
        min_sv = max_sv / dynamic_param
        new_rank = max(torch.sum(S > min_sv).item(), 1)
        new_alpha = float(scale * new_rank)

    elif dynamic_method == "sv_cumulative":
        # Calculate new dim and alpha based off cumulative sum
        new_rank = index_sv_cumulative(S, dynamic_param)
        new_rank = max(new_rank, 1)
        new_alpha = float(scale * new_rank)

    elif dynamic_method == "sv_fro":
        # Calculate new dim and alpha based off sqrt sum of squares
        new_rank = index_sv_fro(S, dynamic_param)
        new_rank = min(max(new_rank, 1), len(S) - 1)
        new_alpha = float(scale * new_rank)
    else:
        new_rank = rank
        new_alpha = float(scale * new_rank)

    if S[0] <= MIN_SV:  # Zero matrix, set dim to 1
        new_rank = 1
        new_alpha = float(scale * new_rank)
    elif new_rank > rank:  # cap max rank at rank
        new_rank = rank
        new_alpha = float(scale * new_rank)

    # Calculate resize info
    s_sum = torch.sum(torch.abs(S))
    s_rank = torch.sum(torch.abs(S[:new_rank]))

    S_squared = S.pow(2)
    s_fro = torch.sqrt(torch.sum(S_squared))
    s_red_fro = torch.sqrt(torch.sum(S_squared[:new_rank]))
    fro_percent = float(s_red_fro / s_fro)

    param_dict["new_rank"] = new_rank
    param_dict["new_alpha"] = new_alpha
    param_dict["sum_retained"] = (s_rank) / s_sum
    param_dict["fro_retained"] = fro_percent
    param_dict["max_ratio"] = S[0] / S[new_rank]

    return param_dict


def extract_linear(weight, lora_rank, dynamic_method, dynamic_param, device, scale=1):
    out_size, in_size = weight.size()

    U, S, Vh = torch.linalg.svd(weight.to(device))

    param_dict = rank_resize(S, lora_rank, dynamic_method, dynamic_param, scale)
    lora_rank = param_dict["new_rank"]

    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S)
    Vh = Vh[:lora_rank, :]

    param_dict["lora_down"] = Vh.reshape(lora_rank, in_size).cpu()
    param_dict["lora_up"] = U.reshape(out_size, lora_rank).cpu()
    del U, S, Vh, weight
    return param_dict


def merge_conv(lora_down, lora_up, device):
    in_rank, in_size, kernel_size, k_ = lora_down.shape
    out_size, out_rank, _, _ = lora_up.shape
    assert in_rank == out_rank and kernel_size == k_, f"rank {in_rank} {out_rank} or kernel {kernel_size} {k_} mismatch"

    lora_down = lora_down.to(device)
    lora_up = lora_up.to(device)

    merged = lora_up.reshape(out_size, -1) @ lora_down.reshape(in_rank, -1)
    weight = merged.reshape(out_size, in_size, kernel_size, kernel_size)
    del lora_up, lora_down
    return weight


def merge_linear(lora_down, lora_up, device):
    in_rank, in_size = lora_down.shape
    out_size, out_rank = lora_up.shape
    assert in_rank == out_rank, f"rank {in_rank} {out_rank} mismatch"

    lora_down = lora_down.to(device)
    lora_up = lora_up.to(device)

    weight = lora_up @ lora_down
    del lora_up, lora_down
    return weight


def str_to_dtype(p):
    if p == 'float':
        return torch.float
    if p == 'fp16':
        return torch.float16
    if p == 'bf16':
        return torch.bfloat16
    return None
