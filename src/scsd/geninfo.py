# Sung-Cheol Kim, Copyright 2023. All rights reserved.

from pydantic import BaseModel


class LDMInfo(BaseModel):
    hfmodel_id: str
    init_index: int
    num_inference_steps: int
    prompt_style: str
    lora_scale: float
    optimize: bool
    prompt: str
    negative_prompt: str
    guidance_scale: float
    enable_compel: bool
    enable_freeu: bool
    height: int
    width: int
    high_gpu_mem: bool
    scheduler: str
    vae: str
    lora_list: list
    use_compel: bool
