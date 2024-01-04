# Sung-Cheol Kim, Copyright 2023. All rights reserved.

import logging
import time
from pathlib import Path
from typing import Dict

import cv2
import tomesd
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, PNDMScheduler
from diffusers.utils import make_image_grid
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LDMHandler")
logger.setLevel(logging.INFO)

torch.backends.cuda.matmul.allow_tf32 = True

model_dict = {"turbo": "stabilityai/sdxl-turbo", "sdxl": "stabilityai/stable-diffusion-xl-base-1.0"}


class LDMHandler(object):
    def __init__(
        self,
        model_base_dir: str = "/home/skim/sd/models",
        sd_output_dir: str = "/mnt/transfer/sd-output",
        mode: str = "turbo",
    ):
        self.model_base_dir = Path(model_base_dir)
        self.sd_output_dir = Path(sd_output_dir)
        self.mode = mode
        self.pipe = None
        self.model_id = ""
        self.init_index = 0
        self.num_inference_steps = 4
        self.prompt_style = ""
        logger.info("output folder: %s", self.sd_output_dir)
        self.set_pipe(mode)

    def _get_inputs(self, prompt: str, batch_size: int = 1) -> Dict:
        generator = [torch.Generator("cuda").manual_seed(i + self.init_index) for i in range(batch_size)]
        prompts = batch_size * [prompt]

        if self.mode == "turbo":
            return {
                "prompt": prompts,
                "generator": generator,
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": 0.0,
            }
        else:
            return {"prompt": prompts, "generator": generator, "num_inference_steps": self.num_inference_steps}

    def set_pipe(self, mode: str = "turbo") -> None:
        start_time = time.time()
        if mode in model_dict:
            self.model_id = model_dict[mode]
        else:
            self.model_id = model_dict["turbo"]
        self.mode = mode

        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )

        # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        # pipe.to("cuda")
        self.pipe.enable_vae_slicing()
        self.pipe.enable_vae_tiling()
        # pipe.enable_xformers_memory_efficient_attention()
        # pipe.enable_sequential_cpu_offload()
        self.pipe.enable_model_cpu_offload()
        self.pipe.unet.to(memory_format=torch.channels_last)
        # tomesd.apply_patch(pipe, ratio=0.5)
        self.pipe.upcast_vae()

        logger.info("model id: %s", self.model_id)
        logger.info("set pipe: %.2f sec", time.time() - start_time)

    def set_scheduler(self, name: str):
        if name == "DPM":
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.num_inference_steps = 20
        elif name == "PNDM":
            self.pipe.scheduler = PNDMScheduler.from_config(self.pipe.scheduler.config)

    def generate(
        self, prompt: str = "", batch_size: int = 8, output: str = "test", mode: str = "", init_index: int = 0
    ):
        start_time = time.time()
        if prompt == "":
            prompt = "A majestic lion jumping from a big stone at night"
        if self.prompt_style != "":
            prompt += self.prompt_style

        if mode != "" and mode != self.mode:
            self.mode = mode
            self.set_pipe(mode)

        if init_index != 0:
            self.init_index = init_index

        inputs = self._get_inputs(prompt, batch_size=batch_size)
        images = self.pipe(**inputs).images

        # create a folder
        (self.sd_output_dir / output).mkdir(exist_ok=True, parents=True)

        for i, img in enumerate(images):
            h, w = img.size
            file_path = self.sd_output_dir / output / f"{output}-H{h}-W{w}-I{i + self.init_index}.jpg"
            if file_path.exists():
                file_path = file_path.parent / file_path.name.replace(".jpg", "-1.jpg")
            img.save(file_path)

        logger.info("Time: %.2f sec", time.time() - start_time)

        if len(images) == 1:
            return images[0]
        elif len(images) == 2:
            return make_image_grid(images, rows=1, cols=2)
        else:
            return make_image_grid(images, rows=2, cols=batch_size // 2)

    def add_style(self, desc: str):
        if desc == "portrait_photo":
            self.prompt_style = (
                " 50mm portrait photography, hard rim lighting photography--beta --ar 2:3 --beta --upbeta"
            )
        else:
            self.prompt_style = ""
