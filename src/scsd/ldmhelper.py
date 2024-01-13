# Sung-Cheol Kim, Copyright 2023. All rights reserved.

import time
import logging
from typing import Dict, List
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
from PIL import Image
from diffusers.utils.pil_utils import make_image_grid

from .utils import grid_images
from .pipeline import SDPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LDMHelper")
logger.setLevel(logging.INFO)

torch.backends.cuda.matmul.allow_tf32 = True


class LDMHelper(object):
    def __init__(
        self,
        model_base_dir: str = "/home/skim/sd/models",
        sd_output_dir: str = "/mnt/transfer/sd-output",
        db_dir: str = "/mnt/transfer/sd-output/sd_db.parquet",
        mode: str = "turbo",
    ):
        self.model_base_dir = Path(model_base_dir)
        self.sd_output_dir = Path(sd_output_dir)
        self.db_path = Path(db_dir)
        logger.info("output folder: %s", self.sd_output_dir)

        self.sd_pipeline = SDPipeline()
        self.sd_pipeline.set_pipeline(mode)

        if not self.db_path.exists():
            self.db = self._create_db()
        else:
            self.db = pd.read_parquet(self.db_path)

        self.mode = mode

    def _create_db(self) -> pd.DataFrame:
        columns = list(self.sd_pipeline.ldm_info.model_fields.keys())
        columns += ["seed", "time", "output_dir", "output_file", "output_file_path"]
        df = pd.DataFrame(columns=columns)
        df.to_parquet(self.db_path)
        return df

    def clean_db(self) -> None:
        idx_list = []
        for i, row in self.db.iterrows():
            p = Path(row["output_file_path"])
            if p.exists():
                idx_list.append(i)
        self.db = self.db.iloc[idx_list].reset_index(drop=True)

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
            self.sd_pipeline.set_pipeline(mode)

        # set prompt
        self.sd_pipeline.set_prompt(prompt)
        if self.sd_pipeline.ldm_info.prompt_style != "":
            self.sd_pipeline.ldm_info.prompt += self.sd_pipeline.ldm_info.prompt_style
        if negative_prompt == "" and self.mode != "turbo":
            self.sd_pipeline.ldm_info.negative_prompt = "ugly, deformed, disfigured, poor details, bad anatomy"
        else:
            self.sd_pipeline.ldm_info.negative_prompt = negative_prompt
        if init_index != 0:
            self.sd_pipeline.ldm_info.init_index = init_index
        if height > 0:
            self.sd_pipeline.ldm_info.height = height
        if width > 0:
            self.sd_pipeline.ldm_info.width = width

        # create database dump and check if the same info exists
        db_dump = self.sd_pipeline.ldm_info.model_dump()
        db_dump["output_dir"] = str(self.sd_output_dir / topic)
        db_dump["lora_list"] = ",".join(self.sd_pipeline.ldm_info.lora_list)

        duplicate_list = self._find_same_info(db_dump)
        images = []
        if len(duplicate_list) >= batch_size:
            for i in range(batch_size):
                for p in duplicate_list:
                    if p["seed"] == i + self.sd_pipeline.ldm_info.init_index:
                        images.append(Image.open(p["output_file_path"]))
                        break

        if len(images) == batch_size:
            logger.info("Loading from cache: %s", self.sd_output_dir / topic)
            return grid_images(images)

        # create images
        images = self.sd_pipeline.generate_batch(batch_size=batch_size)

        # create a folder
        (self.sd_output_dir / topic).mkdir(exist_ok=True, parents=True)

        # save images and update db
        for i, img in enumerate(images):
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            h, w = img.size
            file_path = (
                self.sd_output_dir
                / topic
                / f"H{h}W{w}-{topic}-S{i + self.sd_pipeline.ldm_info.init_index}-{timestamp}.jpg"
            )
            if file_path.exists():
                logger.warning("overwrite file: %s", file_path)
            img.save(file_path)

            db_dump["seed"] = i + self.sd_pipeline.ldm_info.init_index
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

        col_list = [x for x in self.sd_pipeline.ldm_info.model_fields.keys() if x not in ["init_index"]]
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

    def set_default_negative_prompt(self):
        self.negative_prompt = "ugly, deformed, disfigured, poor details, bad anatomy"
