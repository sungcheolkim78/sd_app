# Sung-Cheol Kim, Copyright 2023. All rights reserved.

from importlib import resources as impresources

import yaml
from pydantic import BaseModel

from . import settings


class LDMInfo(BaseModel):
    mode: str
    hfmodel_id: str
    refiner_id: str
    scheduler_id: str
    vae_id: str
    torch_dtype: str
    init_index: int
    num_inference_steps: int
    denoising_end: float
    prompt: str
    negative_prompt: str
    guidance_scale: float
    height: int
    width: int
    prompt_style: str
    lora_scale: float
    lora_list: list
    optimize: bool
    enable_compel: bool
    enable_img2img: bool
    enable_freeu: bool
    high_gpu_mem: bool


def load_ldminfo(ldminfo_path: str) -> LDMInfo:
    with open(ldminfo_path, "r") as f:
        config = yaml.safe_load(f)
    return LDMInfo(**config)


def load_default_ldminfo(mode: str = "turbo") -> LDMInfo:
    config_file = impresources.files(settings) / f"sdxl_{mode}.yaml"

    try:
        with config_file.open("rb") as f:
            config = yaml.safe_load(f)
    except AttributeError:
        text = impresources.read_text(settings, f"sdxl_{mode}.yaml")
        config = yaml.safe_load(text)

    return LDMInfo(**config)
