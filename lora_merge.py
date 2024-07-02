import math
from typing import Literal, get_args

import torch
import comfy

from .peft_utils import task_arithmetic, ties, dare_linear, dare_ties, magnitude_prune, concat
from .utility import find_network_dim, to_dtype

CLAMP_QUANTILE = 0.99
MODES = Literal["add", "concat", "ties", "dare_linear", "dare_ties", "magnitude_prune"]
SVD_MODES = Literal["add_svd", "ties_svd", "dare_linear_svd", "dare_ties_svd", "magnitude_prune_svd"]


class LoraMerger:
    """
       Class for merging LoRA models using various methods.

       Attributes:
           loaded_lora: A placeholder for the loaded LoRA model.
    """

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora1": ("LoRA",),
                "mode": (get_args(MODES),),
                "density": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                }),
                "device": (["cuda", "cpu"],),
                "dtype": (["float32", "float16", "bfloat16"],),
            },
        }

    RETURN_TYPES = ("LoRA",)
    FUNCTION = "lora_merge"
    CATEGORY = "LoRA PowerMerge"

    @torch.no_grad()
    def lora_merge(self, lora1,
                   mode: MODES = None,
                   density=None, device=None, dtype=None, **kwargs):
        """
            Merge multiple LoRA models using the specified mode.

            This method merges the given LoRA models according to the specified mode, such as 'add', 'concat', 'ties',
            'dare_linear', 'dare_ties', or 'magnitude_prune'. The merging process considers the up and down
            projection matrices and their respective alpha values.

            Args:
                lora1 (dict): The first LoRA model to merge.
                mode (str, optional): The merging mode to use. Options include 'add', 'concat', 'ties', 'dare_linear',
                    'dare_ties', and 'magnitude_prune'. Default is None.
                density (float, optional): The density parameter used for some merging modes.
                device (torch.DeviceObjType, optional): The device to use for computations (e.g., 'cuda' or 'cpu').
                dtype (torch.dtype, optional): The data type to use for computations (e.g., 'float32', 'float16', 'bfloat16').
                **kwargs: Additional LoRA models to merge.

            Returns:
                tuple: A tuple containing the merged LoRA model.

            Note:
                - The method ensures that all tensors are moved to the specified device and cast to the specified data type.
                - The merging process involves calculating task weights, scaling with alpha values, and combining
                  up and down projection matrices based on the chosen mode.
        """
        loras = [lora1]
        for k, v in kwargs.items():
            loras.append(v)

        self.validate_input(loras, mode)

        dtype = to_dtype(dtype)
        keys = analyse_keys(loras)
        weight = {}

        # lora = up @ down * alpha / rank
        pbar = comfy.utils.ProgressBar(len(keys))
        for key in keys:
            # Build taskTensor weights
            scale_key = "strength_clip" if "lora_te" in key else "strength_model"
            weights = torch.tensor([w[scale_key] for w in loras]).to(device, dtype=dtype)

            # Calculate up and down nets and their alphas
            ups_downs_alphas = calc_up_down_alphas(loras, key)

            # Scale weights with alpha values
            ups_downs_alphas, alpha_1 = scale_alphas(ups_downs_alphas)

            # Assure that dimensions are equal in every tensor of the same layer
            ups_downs_alphas = curate_tensors(ups_downs_alphas)

            up_tensors = [up.to(device, dtype=dtype) for up, down, alpha in ups_downs_alphas]
            down_tensors = [down.to(device, dtype=dtype) for up, down, alpha in ups_downs_alphas]

            if mode == "add":
                up, down = (task_arithmetic(up_tensors, weights),
                            task_arithmetic(down_tensors, weights))
            elif mode == "concat":
                up, down = (concat(up_tensors, weights, dim=1),
                            concat(down_tensors, weights, dim=0))
            elif mode == "ties":
                up, down = (ties(up_tensors, weights, density),
                            ties(down_tensors, weights, density))
            elif mode == "dare_linear":
                up, down = (dare_linear(up_tensors, weights, density),
                            dare_linear(down_tensors, weights, density))
            elif mode == "dare_ties":
                up, down = (dare_ties(up_tensors, weights, density),
                            dare_ties(down_tensors, weights, density))
            else:  # mode == "magnitude_prune_svd":
                up, down = (magnitude_prune(up_tensors, weights, density),
                            magnitude_prune(down_tensors, weights, density))

            weight[key + ".lora_up.weight"] = up.to('cpu', dtype=torch.float32)
            weight[key + ".lora_down.weight"] = down.to('cpu', dtype=torch.float32)
            weight[key + ".alpha"] = alpha_1.to('cpu', dtype=torch.float32)

            pbar.update(1)

        lora_out = {"lora": weight, "strength_model": 1, "strength_clip": 1,
                    "name": "merged_of_" + "_".join([l['name'] for l in loras])}
        return (lora_out,)

    def validate_input(self, loras, mode):
        dims = [find_network_dim(lora['lora']) for lora in loras]
        if min(dims) != max(dims):
            raise Exception("LoRAs with different ranks not allowed in LoraMerger. Use SVD merge.")
        if mode not in get_args(MODES):
            raise Exception(f"Invalid / unsupported mode {mode}")


class LoraSVDMerger:
    """
        Class for merging LoRA models using Singular Value Decomposition (SVD).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora1": ("LoRA",),
                "mode": (get_args(SVD_MODES),),
                "density": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                }),
                "svd_rank": ("INT", {
                    "default": 16,
                    "min": 1,  # Minimum value
                    "max": 320,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number"  # Cosmetic only: display as "number" or "slider"
                }),
                "svd_conv_rank": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 320,
                    "step": 1,
                    "display": "number"
                }),
                "device": (["cuda", "cpu"],),
                "dtype": (["float32", "float16", "bfloat16"],),
            },
        }

    RETURN_TYPES = ("LoRA",)
    FUNCTION = "lora_svd_merge"
    CATEGORY = "LoRA PowerMerge"

    def lora_svd_merge(self, lora1,
                       mode: SVD_MODES = "add_svd",
                       density: float = None, svd_rank: int = None, svd_conv_rank: int = None, device=None, dtype=None,
                       **kwargs):
        """
           Merge LoRA models using SVD and specified mode.

           Args:
               lora1: The first LoRA model.
               mode: The merging mode to use.
               density: The density parameter for some merging modes.
               svd_rank: The rank for SVD.
               svd_conv_rank: The convolution rank for SVD.
               device: The device to use ('cuda' or 'cpu').
               dtype: The data type for output
               **kwargs: Additional LoRA models to merge.

           Returns:
               A tuple containing the merged LoRA model.
        """
        loras = [lora1]
        for k, v in kwargs.items():
            loras.append(v)
        dtype = to_dtype(dtype)

        self.validate_input(loras, mode)

        weight = {}
        keys = analyse_keys(loras)

        pb = comfy.utils.ProgressBar(len(keys))
        for key in keys:
            # Build taskTensor weights
            strength_key = "strength_clip" if "lora_te" in key else "strength_model"
            strengths = torch.tensor([w[strength_key] for w in loras]).to(device)

            # Calculate up and down nets and their alphas
            ups_downs_alphas = calc_up_down_alphas(loras, key, fill_with_empty_tensor=True)

            # Build merged tensor
            weights = self.build_weights(ups_downs_alphas, strengths, mode, density, device)

            # Calculate final tensors by svd
            up, down, alpha = self.svd(weights, svd_rank, svd_conv_rank, device)

            weight[key + ".lora_up.weight"] = up.to(device='cpu', dtype=dtype)
            weight[key + ".lora_down.weight"] = down.to(device='cpu', dtype=dtype)
            weight[key + ".alpha"] = alpha.to(device='cpu', dtype=dtype)

            pb.update(1)

        lora_out = {"lora": weight, "strength_model": 1, "strength_clip": 1}
        return (lora_out,)

    def validate_input(self, loras, mode):
        if mode not in get_args(SVD_MODES):
            raise Exception(f"Invalid / unsupported mode {mode}")

    def build_weights(self, ups_downs_alphas, strengths,
                      mode: SVD_MODES, density, device):
        """
            Construct the combined weight tensor from multiple LoRA up and down tensors using different
            merging modes.

            This method supports both fully connected (2D) and convolutional (4D) tensors. It scales and
            merges the up and down tensors based on the specified mode and density, performing task-specific
            arithmetic operations.

            Args:
                ups_downs_alphas (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): A list of tuples,
                    where each tuple contains the up tensor, down tensor, and alpha value.
                strengths (torch.Tensor): A tensor containing the strength values for each set of up and down tensors.
                mode (Literal["svd", "ties_svd", "dare_linear_svd", "dare_ties_svd", "magnitude_prune_svd"]):
                    The mode to use for merging the weights. Each mode applies a different method of combining the tensors.
                density (float): The density parameter used in certain modes like "ties" and "dare".
                device (torch.DeviceObjType): The device on which to perform the computations (e.g., 'cuda' or 'cpu').

            Returns:
                torch.Tensor: The combined weight tensor resulting from the specified merging process.

            Note:
                - For convolutional tensors, special handling is applied depending on the kernel size.
                - The weight tensors are scaled by their respective alpha values and normalized by their rank.
        """
        up_1, down_1, alpha_1 = ups_downs_alphas[0]
        conv2d = len(down_1.size()) == 4
        kernel_size = None if not conv2d else down_1.size()[2:4]

        # lora = up @ down * alpha / rank
        weights = []
        for up, down, alpha in ups_downs_alphas:
            up, down, alpha = up.to(device), down.to(device), alpha.to(device)
            rank = up.shape[1]
            if conv2d:
                if kernel_size == (1, 1):
                    weight = (up.squeeze(3).squeeze(2) @ down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(
                        3) * alpha / rank
                else:
                    weight = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3) * alpha / rank
            else:  # linear
                weight = up.view(-1, rank) @ down.view(rank, -1) * alpha / rank
            weights.append(weight)

        if mode == "add_svd":
            weight = task_arithmetic(weights, strengths)
        elif mode == "ties_svd":
            weight = ties(weights, strengths, density)
        elif mode == "dare_linear_svd":
            weight = dare_linear(weights, strengths, density)
        elif mode == "dare_ties_svd":
            weight = dare_ties(weights, strengths, density)
        else:  # mode == "magnitude_prune_svd":
            weight = magnitude_prune(weights, strengths, density)

        return weight

    def svd(self, weights: torch.Tensor, svd_rank: int, svd_conv_rank: int, device: str):
        """
            Perform Singular Value Decomposition (SVD) on the given weights tensor and return the
            decomposed matrices with the specified ranks.

            This method supports both 2D (fully connected) and 4D (convolutional) weight tensors. For
            convolutional tensors, it handles both 1x1 and other kernel sizes. The ranks for decomposition
            are adjusted based on the input tensor's dimensions and the specified rank constraints.

            Args:
                weights (torch.Tensor): The input weight tensor to decompose. Should be either a 2D or 4D tensor.
                svd_rank (int): The rank for SVD decomposition for fully connected layers.
                svd_conv_rank (int): The rank for SVD decomposition for convolutional layers with kernel sizes other than 1x1.
                device (torch.DeviceObjType): The device on which to perform the computations (e.g., 'cuda' or 'cpu').

            Returns:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                    - up_weight: The U matrix after SVD decomposition, representing the left singular vectors.
                    - down_weight: The Vh matrix after SVD decomposition, representing the right singular vectors.
                    - module_new_rank: A tensor containing the new rank used for the decomposition.

            Note:
                SVD only supports float32 data type, so the input weights tensor is converted to float32 if necessary.
        """
        weights = weights.to(dtype=torch.float32, device=device)  # SVD only supports float32

        conv2d = len(weights.size()) == 4
        kernel_size = None if not conv2d else weights.size()[2:4]
        conv2d_3x3 = conv2d and kernel_size != (1, 1)
        out_dim, in_dim = weights.size()[0:2]

        if conv2d:
            if conv2d_3x3:
                weights = weights.flatten(start_dim=1)
            else:
                weights = weights.squeeze()

        module_new_rank = svd_conv_rank if conv2d_3x3 else svd_rank
        module_new_rank = min(module_new_rank, in_dim, out_dim)  # LoRA rank cannot exceed the original dim

        U, S, Vh = torch.linalg.svd(weights)

        U = U[:, :module_new_rank]
        S = S[:module_new_rank]
        U = U @ torch.diag(S)

        Vh = Vh[:module_new_rank, :]

        dist = torch.cat([U.flatten(), Vh.flatten()])
        hi_val = torch.quantile(dist, CLAMP_QUANTILE)
        low_val = -hi_val

        U = U.clamp(low_val, hi_val)
        Vh = Vh.clamp(low_val, hi_val)

        if conv2d:
            U = U.reshape(out_dim, module_new_rank, 1, 1)
            Vh = Vh.reshape(module_new_rank, in_dim, kernel_size[0], kernel_size[1])

        up_weight = U
        down_weight = Vh

        return up_weight, down_weight, torch.tensor(module_new_rank)


@torch.no_grad()
def calc_up_down_alphas(loras, key, fill_with_empty_tensor=False):
    """
       Calculate up, down tensors and alphas for a given key.

       Args:
           loras: List of LoRA models.
           key: The key to calculate values for.
           fill_with_empty_tensor=False: create a zero tensor in the case of a LoRA doesn't contain the key

       Returns:
           List of tuples containing up, down tensors and alpha values.
    """

    up_key = key + ".lora_up.weight"
    down_key = key + ".lora_down.weight"
    alpha_key = key + ".alpha"

    # Find loras with the respective key
    owners = [l for l in loras if down_key in l['lora']]
    up_shape = min([d['lora'][up_key].shape for d in owners])
    down_shape = min([d['lora'][down_key].shape for d in owners])

    # Determine alpha from the first lora which contains the module
    alpha_1 = owners[0]["lora"][alpha_key]

    owner_names = [l["name"] for l in owners]
    out = []
    for lora in loras:
        if lora['name'] in owner_names:
            up, down, alpha = lora["lora"][up_key], lora["lora"][down_key], lora["lora"][alpha_key]
            out.append((up, down, alpha))
        elif fill_with_empty_tensor:
            up, down, alpha = (torch.zeros(up_shape),
                               torch.zeros(down_shape),
                               torch.tensor(alpha_1))
            out.append((up, down, alpha))
    return out


def scale_alphas(ups_downs_alphas):
    up_1, down_1, alpha_1 = ups_downs_alphas[0]
    out = []
    for up, down, alpha in ups_downs_alphas:
        up = up * math.sqrt(alpha / alpha_1)
        down = down * math.sqrt(alpha / alpha_1)
        out.append((up, down, alpha_1))
    return out, alpha_1


def analyse_keys(loras):
    down_keys = set()
    for i, lora in enumerate(loras):
        key_count = 0
        for key in lora["lora"].keys():
            if ".lora_down" in key:
                down_keys.add(key[: key.rfind(".lora_down")])
                key_count += 1
        print(f"LoRA {i} with {key_count} modules.")

    print(f"Total keys to be merged {len(down_keys)} modules")
    return down_keys


def curate_tensors(ups_downs_alphas):
    """
    Checks and eventually curates tensor dimensions
    """
    up_1, down_1, alpha_1 = ups_downs_alphas[0]
    out = [ups_downs_alphas[0]]
    for up, down, alpha in ups_downs_alphas[1:]:
        up = adjust_tensor_to_match(up_1, up)
        down = adjust_tensor_to_match(down_1, down)
        out.append((up, down, alpha))
    return out


def adjust_tensor_to_match(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Adjust tensor2 to match the shape of tensor1.
    If tensor2 is smaller, extend it with zeros.
    If tensor2 is larger, cut it to match the shape of tensor1.

    Args:
        tensor1 (torch.Tensor): The reference tensor with the desired shape.
        tensor2 (torch.Tensor): The tensor to be adjusted.

    Returns:
        torch.Tensor: The adjusted tensor2 matching the shape of tensor1.
    """
    # Get shapes of both tensors
    shape1 = tensor1.shape
    shape2 = tensor2.shape

    # Determine the new shape based on the first tensor
    new_shape = shape1

    # Create a tensor of zeros with the new shape
    adjusted_tensor = torch.zeros(new_shape, dtype=tensor2.dtype)

    # Determine slices for each dimension
    slices = tuple(slice(0, min(dim1, dim2)) for dim1, dim2 in zip(shape1, shape2))

    # Copy the original tensor2 into the adjusted tensor
    adjusted_tensor[slices] = tensor2[slices]

    return adjusted_tensor
