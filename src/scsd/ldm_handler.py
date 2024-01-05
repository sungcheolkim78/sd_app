# Sung-Cheol Kim, Copyright 2023. All rights reserved.

import time
import logging
from typing import Dict, List
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
from PIL import Image
from compel import Compel, ReturnedEmbeddingsType
from diffusers.utils.pil_utils import make_image_grid
from diffusers.pipelines.auto_pipeline import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)

from .utils import grid_images
from .geninfo import LDMInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LDMHandler")
logger.setLevel(logging.INFO)

torch.backends.cuda.matmul.allow_tf32 = True

model_dict = {
    "turbo": "stabilityai/sdxl-turbo",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl-refiner": "stabilityai/stable-diffusion-xl-refiner-1.0",
}


class LDMHandler(object):
    def __init__(
        self,
        model_base_dir: str = "/home/skim/sd/models",
        sd_output_dir: str = "/mnt/transfer/sd-output",
        db_dir: str = "/mnt/transfer/sd-output/sd_db.parquet",
        mode: str = "turbo",
        optimize: bool = True,
    ):
        self.model_base_dir = Path(model_base_dir)
        self.sd_output_dir = Path(sd_output_dir)
        self.db_path = Path(db_dir)
        logger.info("output folder: %s", self.sd_output_dir)

        self.ldm_info = LDMInfo(
            hfmodel_id="",
            init_index=0,
            num_inference_steps=4,
            prompt_style="",
            lora_scale=0.0,
            optimize=optimize,
            prompt="",
            negative_prompt="",
            guidance_scale=0.0,
            enable_compel=True,
            enable_freeu=False,
            height=512,
            width=512,
            high_gpu_mem=True,
            scheduler="",
            vae="",
            lora_list=[],
            use_compel=True,
        )
        if not self.db_path.exists():
            self.db = self._create_db()
        else:
            self.db = pd.read_parquet(self.db_path)

        self.mode = mode
        self.pipe = None
        self.compel = None

    def _create_db(self) -> pd.DataFrame:
        columns = list(self.ldm_info.__fields__.keys())
        columns += ["seed", "time", "output_dir", "output_file", "output_file_path"]
        df = pd.DataFrame(columns=columns)
        df.to_parquet(self.db_path)
        return df

    def clean_db(self):
        idx_list = []
        for i, row in self.db.iterrows():
            p = Path(row["output_file_path"])
            if p.exists():
                idx_list.append(i)
        self.db = self.db.iloc[idx_list].reset_index(drop=True)

    def set_pipe(self, mode: str = "turbo") -> None:
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
            return

        self.ldm_info.hfmodel_id = model_dict[mode]
        if mode == "turbo":
            self.ldm_info.num_inference_steps = 4
            self.ldm_info.guidance_scale = 0.0
            self.ldm_info.height = 512
            self.ldm_info.width = 512
            self.ldm_info.high_gpu_mem = True
            torch_dtype = torch.float16
        elif mode == "sdxl":
            self.ldm_info.num_inference_steps = 20
            self.ldm_info.guidance_scale = 7.5
            self.ldm_info.height = 1024
            self.ldm_info.width = 1024
            self.ldm_info.high_gpu_mem = False
            torch_dtype = torch.bfloat16

        if self.ldm_info.optimize:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                self.ldm_info.hfmodel_id, torch_dtype=torch_dtype, use_safetensors=True, variant="fp16"
            )
        else:
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                self.ldm_info.hfmodel_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )

        # self.pipe_img2img = AutoPipelineForImage2Image.from_pipe(self.pipe)

        self.pipe = self._optimize_pipeline(self.pipe)

        self.compel = Compel(
            tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )

        self.ldm_info.scheduler = self.pipe.scheduler.config._class_name
        self.ldm_info.vae = self.pipe.vae.config._class_name

        logger.info("model id: %s", self.ldm_info.hfmodel_id)
        logger.info("set pipe: %.2f sec", time.time() - start_time)

    def _optimize_pipeline(self, pipe):
        # only for pytorch < 2.0
        # pipe.enable_xformers_memory_efficient_attention()

        pipe.unet.to(memory_format=torch.channels_last)
        pipe.vae.to(memory_format=torch.channels_last)
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        # pipe.upcast_vae()

        if self.ldm_info.high_gpu_mem:
            pipe.to("cuda")
        else:
            pipe.enable_model_cpu_offload()

        return pipe

    def set_scheduler(self, name: str):
        if name == "DPM":
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        elif name == "PNDM":
            self.pipe.scheduler = PNDMScheduler.from_config(self.pipe.scheduler.config)

        self.ldm_info.scheduler = self.pipe.scheduler.config._class_name

    def _get_batch_inputs(self, batch_size: int = 1) -> Dict:
        """
        Get the batch inputs for the LDM handler.

        Args:
            batch_size (int): The size of the batch.

        Returns:
            Dict: A dictionary containing the batch inputs.
        """

        prompts = batch_size * [self.ldm_info.prompt]
        input_dict = {
            "generator": [torch.Generator("cuda").manual_seed(i + self.ldm_info.init_index) for i in range(batch_size)],
            "num_inference_steps": self.ldm_info.num_inference_steps,
            "height": self.ldm_info.height,
            "width": self.ldm_info.width,
        }

        if self.ldm_info.use_compel:
            input_dict["prompt_embeds"], input_dict["pooled_prompt_embeds"] = self.compel(prompts)
        else:
            input_dict["prompt"] = prompts

        if self.ldm_info.lora_scale > 0:
            input_dict["cross_atention_kwargs"] = {"scale": self.ldm_info.lora_scale}

        if self.mode == "turbo":
            input_dict["guidance_scale"] = self.ldm_info.guidance_scale

        if self.ldm_info.negative_prompt != "":
            input_dict["negative_prompt"] = batch_size * [self.ldm_info.negative_prompt]

        info_dict = {
            "prompt": self.ldm_info.prompt,
            "negative_prompt": self.ldm_info.negative_prompt,
            "batch_size": batch_size,
            "num_inference_steps": self.ldm_info.num_inference_steps,
            "guidance_scale": self.ldm_info.guidance_scale,
        }
        logger.info("input_dict: %s", info_dict)

        return input_dict

    def txt2img(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        batch_size: int = 8,
        topic: str = "test",
        mode: str = "",
        init_index: int = 0,
        height: int = -1,
        width: int = -1,
    ):
        start_time = time.time()

        # set mode
        if mode != "" and mode != self.mode:
            self.mode = mode
            self.set_pipe(mode)
        elif self.pipe is None:
            self.set_pipe(self.mode)

        # set prompt
        self.ldm_info.prompt = prompt
        if self.ldm_info.prompt_style != "":
            self.ldm_info.prompt += self.ldm_info.prompt_style
        if negative_prompt == "" and self.mode != "turbo":
            self.ldm_info.negative_prompt = "ugly, deformed, disfigured, poor details, bad anatomy"
        else:
            self.ldm_info.negative_prompt = negative_prompt
        if init_index != 0:
            self.ldm_info.init_index = init_index
        if height > 0:
            self.ldm_info.height = height
        if width > 0:
            self.ldm_info.width = width

        # create database dump and check if the same info exists
        db_dump = self.ldm_info.model_dump()
        db_dump["output_dir"] = str(self.sd_output_dir / topic)
        db_dump["lora_list"] = ",".join(self.ldm_info.lora_list)

        duplicate_list = self._find_same_info(db_dump)
        images = []
        if len(duplicate_list) >= batch_size:
            for i in range(batch_size):
                for p in duplicate_list:
                    if p["seed"] == i + self.ldm_info.init_index:
                        images.append(Image.open(p["output_file_path"]))
                        break

        if len(images) == batch_size:
            logger.info("Loading from cache: %s", self.sd_output_dir / topic)
            return grid_images(images)

        # create batch inputs
        inputs = self._get_batch_inputs(batch_size=batch_size)

        # create images
        images = self.pipe(**inputs).images

        # create a folder
        (self.sd_output_dir / topic).mkdir(exist_ok=True, parents=True)

        # save images and update db
        for i, img in enumerate(images):
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            h, w = img.size
            file_path = self.sd_output_dir / topic / f"H{h}W{w}-{topic}-S{i + self.ldm_info.init_index}-{timestamp}.jpg"
            if file_path.exists():
                logger.warning("overwrite file: %s", file_path)
            img.save(file_path)

            db_dump["seed"] = i + self.ldm_info.init_index
            db_dump["output_file"] = file_path.name
            db_dump["output_file_path"] = str(file_path)
            db_dump["time"] = timestamp
            df_from_dict = pd.DataFrame.from_dict([db_dump])
            self.db = pd.concat([self.db, df_from_dict], ignore_index=True)

        self.db.to_parquet(self.db_path)
        logger.info("Time: %.2f sec, Write to: %s", time.time() - start_time, self.db_path)

        return grid_images(images)

    def _find_same_info(self, db_dump: Dict) -> List:
        """
        Find the same info from the database.

        Args:
            db_dump (Dict): The database dump.

        Returns:
            List: A list of the same info.
        """

        col_list = [x for x in self.ldm_info.__fields__.keys() if x not in ["init_index"]]
        same_info = []
        for i, row in self.db.iterrows():
            for col in col_list:
                if row[col] != db_dump[col]:
                    break
            else:
                same_info.append({"seed": row["seed"], "output_file_path": row["output_file_path"]})

        return same_info

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

    def add_style(self, desc: str):
        if desc == "portrait_photo":
            self.ldm_info.prompt_style = (
                " 50mm portrait photography, hard rim lighting photography--beta --ar 2:3 --beta --upbeta"
            )
        else:
            self.ldm_info.prompt_style = ""

    def load_lora_toy(self):
        lora_id = "CiroN2022/toy-face"
        lora_filename = "toy_face_sdxl.safetensors"
        lora_name = "toy"

        active_adapters = self.pipe.get_active_adapters()
        if lora_name in active_adapters:
            logger.warning("%s is already loaded!", lora_name)
            return

        self.pipe.load_lora_weights(lora_id, weight_name=lora_filename, adapter_name=lora_name)
        self.ldm_info.lora_scale = 0.9
        self.ldm_info.lora_list.append(lora_name)
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
        self.ldm_info.lora_scale = 0.9
        self.ldm_info.lora_list.append(lora_name)
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
        self.ldm_info.lora_scale = 0.0
        self.ldm_info.lora_list = []

    def compile(self, mode: str = "reduce-overhead"):
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
        self.pipe.unet.to(memory_format=torch.channels_last)
        self.pipe.vae.to(memory_format=torch.channels_last)

        self.pipe.unet = torch.compile(self.pipe.unet, mode=mode, fullgraph=True)
        self.pipe.vae.decode = torch.compile(self.pipe.vae.decode, mode=mode, fullgraph=True)

        logger.info("compile model: %.2f sec", time.time() - start_time)

    def set_default_negative_prompt(self):
        self.negative_prompt = "ugly, deformed, disfigured, poor details, bad anatomy"

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

    def refine(self, prompt: str = "", num_inference_steps: int = 40, denoising_start: float = 0.8):
        # load model
        # self.ldm_info.use_compel = False
        self.ldm_info.high_gpu_mem = False
        self.ldm_info.hfmodel_id = model_dict["sdxl"]
        self.pipe = DiffusionPipeline.from_pretrained(
            self.ldm_info.hfmodel_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )
        self.pipe = self._optimize_pipeline(self.pipe)

        self.refiner = DiffusionPipeline.from_pretrained(
            model_dict["sdxl-refiner"],
            text_encoder_2=self.pipe.text_encoder_2,
            vae=self.pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.refiner = self._optimize_pipeline(self.refiner)

        # create image
        self.ldm_info.prompt = prompt
        self.ldm_info.num_inference_steps = num_inference_steps
        self.ldm_info.height = 1024
        self.ldm_info.width = 1024
        batch_input = self._get_batch_inputs(batch_size=1)
        batch_input["denoising_end"] = denoising_start
        batch_input["output_type"] = "latent"
        image = self.pipe(**batch_input).images

        image = self.refiner(
            prompt=self.ldm_info.prompt,
            num_inference_steps=num_inference_steps,
            denoising_start=denoising_start,
            image=image,
        ).images[0]

        return image
