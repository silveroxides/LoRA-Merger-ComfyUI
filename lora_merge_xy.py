
import numpy as np
import os

import folder_paths
import nodes

from .lora_merge import LoraMerger
from .lora_load import LoraPowerMergeLoader

from comfy.sd import load_lora_for_models


class XYInputLoraMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_a": (folder_paths.get_filename_list("loras"), ),
                "lora_b": (folder_paths.get_filename_list("loras"), ),
                "mode": (["add", "ties", "dare_linear", "dare_ties", "magnitude_prune"], ),
                "density": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                }),
                "device": (["cuda", "cpu"],),
                "dtype": (["float32", "float16", "bfloat16"],),
                "min_strength": ("FLOAT", {"default": 0, "min": -0.0, "max": 1.0, "step": 0.01}),
                "max_strength": ("FLOAT", {"default": 1.0, "min":  0.0, "max": 1.0, "step": 0.01}),
                "apply_strength": (["model + clip", "model", "clip"],),
                "steps": ("INT", {"default": 4, "min": 0, "max": 320, "step": 1, "display": "number"}),
            }
        }

    RETURN_TYPES = ("XY", "XY")
    RETURN_NAMES = ("X (vectors)", "Y (effect_compares)")

    FUNCTION = "doit"
    CATEGORY = "LoRA PowerMerge"

    def doit(self, lora_a, lora_b, mode: str,  density, device, dtype,
             min_strength: float, max_strength: float, apply_strength, steps):
        strength_values = generate_floats(steps, min_strength, max_strength)
        storage = {}
        x_values, y_values = [], []
        common_params = lora_a, lora_b, mode, density, device, dtype, apply_strength

        for x, sv1 in enumerate(strength_values):
            x_values.append(XYLoRAMergeCapsule(x, 0, strength_values, f'{os.path.splitext(lora_a)[0][-30:]}:{sv1:.2f}', storage, common_params))

        for y, sv2 in enumerate(strength_values):
            y_values.append(XYLoRAMergeCapsule(0, y, strength_values, f'{os.path.splitext(lora_b)[0][-30:]}:{sv2:.2f}', storage, common_params))

        xy_type = "XY_Capsule"
        return (xy_type, x_values), (xy_type, y_values)


def generate_floats(batch_count, first_float, last_float):
    if batch_count > 1:
        interval = (last_float - first_float) / (batch_count - 1)
        return [round(first_float + i * interval, 3) for i in range(batch_count)]
    else:
        return [first_float] if batch_count == 1 else []

class XYLoRAMergeCapsule:
    def __init__(self, x, y, strengths, label, storage, params):
        self.x = x
        self.y = y
        self.strengths = strengths
        self.label = label
        self.storage = storage
        self.params = params

    def pre_define_model(self, model, clip, vae):
        model, clip = self.patch_model(model, clip)
        return model, clip, vae

    def patch_model(self, model, clip):
        merger = LoraMerger()
        lora_a, lora_b, mode, density, device, dtype, apply_strength = self.params

        loader = LoraPowerMergeLoader()
        x_model_strength = self.strengths[self.another_capsule.x] if apply_strength != 'clip' else 0
        x_clip_strength = self.strengths[self.another_capsule.x] if apply_strength != 'model' else 0
        y_model_strength = self.strengths[self.y] if apply_strength != 'clip' else 0
        y_clip_strength = self.strengths[self.y] if apply_strength != 'model' else 0

        lora_a = loader.load_lora(lora_a, x_model_strength, x_clip_strength, "")[0]
        lora_b = loader.load_lora(lora_b, y_model_strength, y_clip_strength, "")[0]

        lora = merger.lora_merge(lora_a, mode, density, device, dtype, lora_b=lora_b)
        model_lora, clip_lora = load_lora_for_models(model, clip, lora[0]['lora'], 1, 1)
        return model_lora, clip_lora

    def getLabel(self):
        # override
        return self.label

    def set_x_capsule(self, capsule):
        self.another_capsule = capsule

    def set_result(self, image, latent):
        if self.another_capsule is not None:
            print(f"XY_Capsule_LoraBlockWeight: ({self.another_capsule.x, self.y}) is processed.")
            self.storage[(self.another_capsule.x, self.y)] = image
        else:
            print(f"XY_Capsule_LoraBlockWeight: ({self.x, self.y}) is processed.")

    def get_result(self, model, clip, vae):
        if (self.another_capsule.x, self.y) not in self.storage:
            return None

        image = self.storage[(self.another_capsule.x, self.y)]
        latent = nodes.VAEEncode().encode(vae, image)[0]
        return (image, latent)


