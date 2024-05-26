from .lora_merge import LoraMerger, LoraSVDMerger, LoraResizer
from .lora_load_from_weight import LoraLoaderFromWeight
from .lora_load_weight_only import LoraLoaderWeightOnly
from .lora_save import LoraSave

version_code = [0, 10]
version_str = f"V{version_code[0]}.{version_code[1]}" + (f'.{version_code[2]}' if len(version_code) > 2 else '')
print(f"### Loading: ComfyUI-LoRA-Merge ({version_str})")

NODE_CLASS_MAPPINGS = {
    "LoraMerger": LoraMerger,
    "LoraSVDMerger": LoraSVDMerger,
    "LoraResizer": LoraResizer,
    "LoraLoaderFromWeight": LoraLoaderFromWeight,
    "LoraLoaderWeightOnly": LoraLoaderWeightOnly,
    "LoraSave": LoraSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraMerger": "Merge LoRA",
    "LoraSVDMerger": "Merge LoRA SVD",
    "LoraResizer": "Resize LoRA",
    "LoraLoaderFromWeight": "Load LoRA from Weight",
    "LoraLoaderWeightOnly": "Load LoRA Weight Only",
    "LoraSave": "Save LoRA",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


