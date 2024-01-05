# Sung-Cheol Kim, Copyright 2023. All rights reserved.

import time
import logging
from typing import Dict
from pathlib import Path

import torch
from PIL import Image
from diffusers import (
    PNDMScheduler,
    AutoPipelineForText2Image,
    StableDiffusionXLPipeline,
    AutoPipelineForImage2Image,
    DPMSolverMultistepScheduler,
)
from diffusers.utils import make_image_grid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LDMHandler")
logger.setLevel(logging.INFO)

torch.backends.cuda.matmul.allow_tf32 = True

# for model compile - Currently it does not work well.
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

model_dict = {"turbo": "stabilityai/sdxl-turbo", "sdxl": "stabilityai/stable-diffusion-xl-base-1.0"}


class LDMHandler(object):
    def __init__(
        self,
        model_base_dir: str = "/home/skim/sd/models",
        sd_output_dir: str = "/mnt/transfer/sd-output",
        mode: str = "turbo",
        optimize: bool = True,
    ):
        self.model_base_dir = Path(model_base_dir)
        self.sd_output_dir = Path(sd_output_dir)
        logger.info("output folder: %s", self.sd_output_dir)

        self.mode = mode
        self.pipe = None
        self.model_id = ""
        self.init_index = 0
        self.num_inference_steps = 4
        self.prompt_style = ""
        self.lora_scale = 0.0
        self.optimize = optimize
        self.prompt = ""
        self.negative_prompt = ""
        self.set_pipe(mode)

    def set_pipe(self, mode: str = "turbo") -> None:
        start_time = time.time()
        if mode in model_dict:
            self.model_id = model_dict[mode]
            if mode == "turbo":
                self.num_inference_steps = 4
                self.height = 512
                self.width = 512
            else:
                self.num_inference_steps = 20
            self.mode = mode
        else:
            logger.warning("invalid mode: %s", mode)
            return

        if self.optimize:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_id, torch_dtype=torch.bfloat16, use_safetensors=True, variant="fp16"
            )
        else:
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )

        self.pipe_img2img = AutoPipelineForImage2Image.from_pipe(self.pipe)

        # pipe.enable_xformers_memory_efficient_attention()
        # pipe.enable_sequential_cpu_offload()
        # tomesd.apply_patch(pipe, ratio=0.5)
        self.pipe.unet.to(memory_format=torch.channels_last)
        self.pipe.vae.to(memory_format=torch.channels_last)
        self.pipe.enable_vae_slicing()
        self.pipe.enable_vae_tiling()
        # self.pipe.enable_model_cpu_offload()
        self.pipe.to("cuda")
        self.pipe.upcast_vae()

        logger.info("model id: %s", self.model_id)
        logger.info("set pipe: %.2f sec", time.time() - start_time)

    def set_scheduler(self, name: str):
        if name == "DPM":
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.num_inference_steps = 20
        elif name == "PNDM":
            self.pipe.scheduler = PNDMScheduler.from_config(self.pipe.scheduler.config)

    def _get_inputs(self, prompt: str, negative_prompt: str = "", batch_size: int = 1) -> Dict:
        input_dict = {
            "prompt": batch_size * [prompt],
            "generator": [torch.Generator("cuda").manual_seed(i + self.init_index) for i in range(batch_size)],
            "num_inference_steps": self.num_inference_steps,
        }

        if self.lora_scale > 0:
            input_dict["cross_atention_kwargs"] = {"scale": self.lora_scale}

        if self.mode == "turbo":
            input_dict["guidance_scale"] = 0.0

        if negative_prompt != "":
            input_dict["negative_prompt"] = batch_size * [negative_prompt]

        logger.info("input_dict: %s", input_dict)
        return input_dict

    def generate(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        batch_size: int = 8,
        output: str = "test",
        mode: str = "",
        init_index: int = 0,
        height: int = -1,
        width: int = -1,
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

        inputs = self._get_inputs(prompt, negative_prompt=negative_prompt, batch_size=batch_size)
        images = self.pipe(**inputs).images

        # create a folder
        (self.sd_output_dir / output).mkdir(exist_ok=True, parents=True)

        for i, img in enumerate(images):
            h, w = img.size
            file_path = self.sd_output_dir / output / f"{output}-H{h}-W{w}-I{i + self.init_index}.jpg"
            while file_path.exists():
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

    def img2img(self, prompt: str = "", output: str = "test", index: int = 0):
        for p in (self.sd_output_dir / output).glob(f"{output}*-I{index + 1}.jpg"):
            logger.info("input file: %s", p)
            image = Image.open(p).convert("RGB")
            image.thumbnail((512, 512))

            new_image = self.pipe_img2img(
                prompt, image=image, num_inference_steps=4, strength=0.5, guidance_scale=0.0
            ).images[0]
            break

        return make_image_grid([image, new_image], rows=1, cols=2)

    def load_lora_toy(self):
        lora_id = "CiroN2022/toy-face"
        lora_filename = "toy_face_sdxl.safetensors"
        lora_name = "toy"

        active_adapters = self.pipe.get_active_adapters()
        if lora_name in active_adapters:
            logger.warning("%s is already loaded!", lora_name)
            return

        self.pipe.load_lora_weights(lora_id, weight_name=lora_filename, adapter_name=lora_name)
        self.lora_scale = 0.9
        logger.info("lora: %s name: %s", lora_id, lora_name)

    def load_lora_pixel(self):
        lora_id = "nerijs/pixel-art-xl"
        lora_filename = "pixel-art-xl.safetensors"
        lora_name = "pixel"

        active_adapters = self.pipe.get_active_adapters()
        if lora_name in active_adapters:
            logger.warning("%s is already loaded!", lora_name)
            return

        self.pipe.load_lora_weights(lora_id, weight_name=lora_filename, adapter_name=lora_name)
        self.lora_scale = 0.9
        logger.info("lora: %s name: %s", lora_id, lora_name)

    def load_lora_toy_pixel(self):
        active_adapters = self.pipe.get_active_adapters()
        if "toy" not in active_adapters:
            self.load_lora_toy()
        if "pixel" not in active_adapters:
            self.load_lora_pixel()
        self.pipe.set_adapters(["pixel", "toy"], adapter_weights=[0.5, 1.0])

    def disable_lora(self):
        self.pipe.disable_lora()
        self.lora_scale = 0.0

    def compile(self, mode: str = "reduce-overhead"):
        """compile unet and vae for speed-up
        Args:
            mode: ["reduce-overhead", "max-autotune"]
        """

        start_time = time.time()
        self.pipe.unet.to(memory_format=torch.channels_last)
        self.pipe.vae.to(memory_format=torch.channels_last)

        self.pipe.unet = torch.compile(self.pipe.unet, mode=mode, fullgraph=True)
        self.pipe.vae.decode = torch.compile(self.pipe.vae.decode, mode=mode, fullgraph=True)

        logger.info("compile model: %.2f sec", time.time() - start_time)

    def set_default_negative_prompt(self):
        self.negative_prompt = "ugly, deformed, disfigured, poor details, bad anatomy"
