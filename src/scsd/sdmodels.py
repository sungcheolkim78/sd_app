# Sung-Cheol Kim, Copyright 2023. All rights reserved.

import logging
import time
from typing import Dict

import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers.pipelines.auto_pipeline import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
)

from .geninfo import LDMInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

model_dict = {
    "turbo": "stabilityai/sdxl-turbo",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "refiner": "stabilityai/stable-diffusion-xl-refiner-1.0",
}


def set_pipe(ldm_info: LDMInfo, mode: str = "turbo") -> Dict:
    """
    Sets the pipeline for the LDM handler based on the specified mode.

    Args:
        mode (str, optional): The mode to set the pipeline. Defaults to "turbo".

    Returns:
        None
    """

    start_time = time.time()

    if mode not in model_dict:
        logger.warning("invalid mode: %s", mode)
        return {}

    ldm_info.hfmodel_id = model_dict[mode]
    if mode == "turbo":
        ldm_info.num_inference_steps = 4
        ldm_info.guidance_scale = 0.0
        ldm_info.height = 512
        ldm_info.width = 512
        ldm_info.high_gpu_mem = True
        torch_dtype = torch.float16
    elif mode == "sdxl":
        ldm_info.num_inference_steps = 20
        ldm_info.guidance_scale = 7.5
        ldm_info.height = 1024
        ldm_info.width = 1024
        ldm_info.high_gpu_mem = False
        torch_dtype = torch.bfloat16

    torch_dtype = torch.float16 if ldm_info.high_gpu_mem else torch.bfloat16

    pipe = AutoPipelineForText2Image.from_pretrained(
        ldm_info.hfmodel_id,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant="fp16",
    )

    pipe_img2img = AutoPipelineForImage2Image.from_pipe(pipe)
    refiner = None

    pipe = _optimize_pipeline(pipe, ldm_info)

    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
    )

    ldm_info.scheduler = pipe.scheduler.config._class_name
    ldm_info.vae = pipe.vae.config._class_name

    logger.info("model id: %s", ldm_info.hfmodel_id)
    logger.info("set pipe: %.2f sec", time.time() - start_time)

    return {
        "pipeline": pipe,
        "pipeline_img2img": pipe_img2img,
        "refiner": refiner,
        "compel": compel,
        "ldm_info": ldm_info,
    }


def _optimize_pipeline(pipe, ldm_config):
    # only for pytorch < 2.0
    # pipe.enable_xformers_memory_efficient_attention()

    pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    # pipe.upcast_vae()

    if ldm_config.high_gpu_mem:
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()

    return pipe
