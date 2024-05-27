from .lora_apply import LoraApply
from .lora_load import LoraPowerMergeLoader
from .lora_merge import LoraMerger, LoraSVDMerger
from .lora_resize import LoraResizer
from .lora_merge_xy import XYInputLoraMerge
from .lora_save import LoraSave

version_code = [0, 10]
version_str = f"V{version_code[0]}.{version_code[1]}" + (f'.{version_code[2]}' if len(version_code) > 2 else '')
print(f"### Loading: ComfyUI LoRA-PowerMerge ({version_str})")

NODE_CLASS_MAPPINGS = {
    "LoraMerger": LoraMerger,
    "LoraSVDMerger": LoraSVDMerger,
    "LoraResizer": LoraResizer,
    "LoraApply": LoraApply,
    "LoraPowerMergeLoader": LoraPowerMergeLoader,
    "LoraSave": LoraSave,
    "XYInputLoraMerge": XYInputLoraMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraMerger": "Merge LoRA",
    "LoraSVDMerger": "Merge LoRA SVD",
    "LoraResizer": "Resize LoRA",
    "LoraApply": "Apply LoRA",
    "LoraPowerMergeLoader": "Load LoRA (PowerMerge)",
    "LoraSave": "Save LoRA",
    "XYInputLoraMerge": "XY: LoRA Merge",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
