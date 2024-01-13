# Sung-Cheol Kim, Copyright 2023. All rights reserved.

import time
import logging
from typing import Dict, List

import torch
from PIL import Image
from compel import Compel, ReturnedEmbeddingsType
from diffusers.pipelines.auto_pipeline import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)

from .schema import LDMInfo, load_default_ldminfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SDPipeline(object):
    def __init__(self) -> None:
        self.ldm_info = None
        self.pipe = None
        self.pipe_img2img = None
        self.refiner = None
        self.compel = None

    def set_pipeline(self, mode: str = "turbo"):
        if self.ldm_info is not None and self.ldm_info.mode == mode:
            logger.info("mode is already set: %s", mode)
            return

        logger.info("load default ldm_info: %s", mode)
        self.ldm_info = load_default_ldminfo(mode)

        # clean up GPU memory
        self.pipe = None
        self.pipe_img2img = None
        self.refiner = None
        self.compel = None
        torch.cuda.empty_cache()

        output = load_components(self.ldm_info)
        self.pipe = output["pipeline"]
        self.pipe_img2img = output["pipeline_img2img"]
        self.refiner = output["refiner"]
        self.compel = output["compel"]

    def set_prompt(self, prompt: str):
        self.ldm_info.prompt = prompt

    def set_scheduler(self, name: str):
        if name == "DPM":
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        elif name == "PNDM":
            self.pipe.scheduler = PNDMScheduler.from_config(self.pipe.scheduler.config)

        self.ldm_info.scheduler_id = self.pipe.scheduler.config._class_name

    def generate_batch(self, batch_size: int = 4) -> List[Image.Image]:
        input_dict = self._get_batch_inputs(batch_size=batch_size)
        images = self._generate_image_from_text(input_dict)
        return images

    def _get_single_inputs(self) -> Dict:
        return self.get_batch_inputs(batch_size=1)

    def _get_batch_inputs(self, batch_size: int = 4) -> Dict:
        input_dict = {
            "generator": [torch.Generator("cuda").manual_seed(i + self.ldm_info.init_index) for i in range(batch_size)],
            "num_inference_steps": self.ldm_info.num_inference_steps,
            "height": self.ldm_info.height,
            "width": self.ldm_info.width,
            "guidance_scale": self.ldm_info.guidance_scale,
        }

        prompts = batch_size * [self.ldm_info.prompt]
        if self.ldm_info.enable_compel:
            input_dict["prompt_embeds"], input_dict["pooled_prompt_embeds"] = self.compel(prompts)
        else:
            input_dict["prompt"] = prompts

        if self.ldm_info.lora_scale > 0:
            input_dict["cross_atention_kwargs"] = {"scale": self.ldm_info.lora_scale}

        if self.ldm_info.negative_prompt != "":
            input_dict["negative_prompt"] = [self.ldm_info.negative_prompt]

        info_dict = {
            "prompt": self.ldm_info.prompt,
            "negative_prompt": self.ldm_info.negative_prompt,
            "initial seed": self.ldm_info.init_index,
            "batch_size": batch_size,
            "num_inference_steps": self.ldm_info.num_inference_steps,
            "guidance_scale": self.ldm_info.guidance_scale,
        }
        logger.info("input_dict: %s", info_dict)

        return input_dict

    def _generate_image_from_text(self, input_dict: Dict) -> List[Image.Image]:
        if self.ldm_info.mode == "refine":
            input_dict["denoising_end"] = self.ldm_info.denoising_end
            input_dict["output_type"] = "latent"

            if "prompt_embeds" in input_dict:
                batch_size = len(input_dict["prompt_embeds"])
            else:
                batch_size = len(input_dict["prompt"])

            refine_dict = {
                "prompt": batch_size * [self.ldm_info.prompt],
                "num_inference_steps": self.ldm_info.num_inference_steps,
                "denoising_start": self.ldm_info.denoising_end,
            }

            refine_dict["image"] = self.pipe(**input_dict).images
            images = self.refiner(**refine_dict).images
        else:
            images = self.pipe(**input_dict).images

        return images

    def generate_image_from_image(self, input_dict):
        pass

    def compile(self, mode: str = "reduce-overhead") -> None:
        """compile unet and vae for speed-up
        Args:
            mode: ["reduce-overhead", "max-autotune"]
        """

        # for model compile - Currently it does not work well.
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True

        start_time = time.time()

        self.pipe.unet = torch.compile(self.pipe.unet, mode=mode, fullgraph=True)
        self.pipe.vae.decode = torch.compile(self.pipe.vae.decode, mode=mode, fullgraph=True)

        logger.info("compile model: %.2f sec", time.time() - start_time)

    def enable_freeu(self):
        if "stable-diffusion-v1-5" in self.ldm_info.hfmodel_id:
            self.pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
        elif "stable-diffusion-xl" in self.ldm_info.hfmodel_id:
            # https://wandb.ai/nasirk24/UNET-FreeU-SDXL/reports/FreeU-SDXL-Optimal-Parameters--Vmlldzo1NDg4NTUw
            self.pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        elif "sdxl-turbo" in self.ldm_info.hfmodel_id:
            # https://wandb.ai/nasirk24/UNET-FreeU-SDXL/reports/FreeU-SDXL-Optimal-Parameters--Vmlldzo1NDg4NTUw
            self.pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)

        self.ldm_info.enable_freeu = True

    def disable_freeu(self):
        self.pipe.disable_freeu()
        self.ldm_info.enable_freeu = False


def load_components(ldm_info: LDMInfo) -> Dict:
    """
    Sets the pipeline for the LDM handler based on the specified mode.

    Args:
        mode (str, optional): The mode to set the pipeline. Defaults to "turbo".

    Returns:
        None
    """

    start_time = time.time()

    torch_dtype_dict = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = torch_dtype_dict[ldm_info.torch_dtype]
    logger.info("torch dtype: %s", ldm_info.torch_dtype)

    output = {}
    if ldm_info.vae_id != "" and ldm_info.vae_id != "AutoencoderKL":
        vae = AutoencoderKL.from_pretrained(ldm_info.vae_id, torch_dtype=torch_dtype)
    else:
        vae = AutoencoderKL.from_pretrained(ldm_info.hfmodel_id, subfolder="vae", torch_dtype=torch_dtype)
    logger.info("vae id: %s", ldm_info.vae_id)

    if ldm_info.hfmodel_id != "":
        output["pipeline"] = DiffusionPipeline.from_pretrained(
            ldm_info.hfmodel_id,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant="fp16",
            vae=vae,
        )
        pipe = output["pipeline"]
        pipe = _optimize_pipeline(pipe, ldm_info)
    else:
        raise ValueError("hfmodel_id is empty")
    logger.info("hfmodel id: %s", ldm_info.hfmodel_id)

    if ldm_info.refiner_id != "":
        output["refiner"] = DiffusionPipeline.from_pretrained(
            ldm_info.refiner_id,
            text_encoder_2=pipe.text_encoder_2,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant="fp16",
            vae=vae,
        )
        output["refiner"] = _optimize_pipeline(output["refiner"], ldm_info)
    else:
        output["refiner"] = None
    logger.info("refiner id: %s", ldm_info.refiner_id)

    if ldm_info.enable_img2img:
        output["pipeline_img2img"] = AutoPipelineForImage2Image.from_pipe(pipe)
    else:
        output["pipeline_img2img"] = None
    logger.info("img2img: %s", ldm_info.enable_img2img)

    if ldm_info.enable_compel:
        output["compel"] = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )
    else:
        output["compel"] = None
    logger.info("compel: %s", ldm_info.enable_compel)

    ldm_info.scheduler_id = pipe.scheduler.config._class_name
    ldm_info.vae_id = pipe.vae.config._class_name

    logger.info("load pipeline: %.2f sec", time.time() - start_time)

    return output


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
