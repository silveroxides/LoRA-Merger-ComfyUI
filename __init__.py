from .lora_apply import LoraApply
from .lora_load import LoraPowerMergeLoader
from .lora_merge import LoraMerger, LoraSVDMerger
from .lora_resize import LoraResizer
from .lora_merge_xy import XYInputPowerMergeStrengths, XYInputPowerMergeModes, XYInputPowerMergeSVD
from .lora_save import LoraSave

version_code = [0, 11]
version_str = f"V{version_code[0]}.{version_code[1]}" + (f'.{version_code[2]}' if len(version_code) > 2 else '')
print(f"### Loading: ComfyUI LoRA-PowerMerge ({version_str})")

NODE_CLASS_MAPPINGS = {
    "PM LoRA Merger": LoraMerger,
    "PM LoRA SVD Merger": LoraSVDMerger,
    "PM LoRA Resizer": LoraResizer,
    "PM LoRA Apply": LoraApply,
    "PM LoRA Loader": LoraPowerMergeLoader,
    "PM LoRA Save": LoraSave,
    "XY: PM LoRA Strengths": XYInputPowerMergeStrengths,
    "XY: PM LoRA Modes": XYInputPowerMergeModes,
    "XY: PM LoRA SVD Rank": XYInputPowerMergeSVD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PM LoRA Merger": "PM Merge LoRA",
    "PM LoRA SVD Merger": "PM Merge LoRA SVD",
    "PM LoRA Resizer": "PM Resize LoRA",
    "PM LoRA Apply": "PM Apply LoRA",
    "PM LoRA Loader": "PM Load LoRA",
    "PM LoRA Save": "PM Save LoRA",
    "XY: PM LoRA Strengths": "XY: LoRA Power-Merge Strengths",
    "XY: PM LoRA Modes": "XY: LoRA Power-Merge Modes",
    "XY: PM LoRA SVD Rank": "XY: LoRA Power-Merge SVD Rank",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
