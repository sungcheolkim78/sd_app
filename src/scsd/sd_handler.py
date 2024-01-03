# Sung-Cheol Kim, Copyright 2024. All rights reserved.

import time
import logging
import subprocess
from pathlib import Path

from PIL import Image

logger = logging.getLogger("SDHandler")
logger.setLevel(logging.INFO)


class SDHandler(object):
    def __init__(self, model_base_dir: str = "/home/skim/sd/models", sd_output_dir: str = "/home/skim/sd/output"):
        self.model_base_dir = Path(model_base_dir)
        self.output_dir = Path(sd_output_dir)
        self.sd_cmd = "sd"
        self.sd_dict = {
            "mode": "txt2img",
            "threads": -1,
            "model": "",
            "vae": "",
            "taesd": "",
            "type": "f32",
            "lora-model-dir": model_base_dir,
            "init-img": "",
            "output": "",
            "prompt": "",
            "negative-prompt": "",
            "cfg-scale": 7.0,
            "strength": 0.75,
            "height": 512,
            "width": 512,
            "sampling-method": "euler_a",
            "steps": 20,
            "seed": -1,
            "batch-count": 1,
            "schedule": "discrete",
            "clip-skip": -1,
        }

    def set_sdxl(self):
        self.sd_dict.update(
            {
                "model": str(self.model_base_dir / "sd_xl_base_1.0.safetensors"),
                "vae": str(self.model_base_dir / "sdxl_vae.safetensors"),
                "height": 1024,
                "width": 1024,
                "type": "f16",
            }
        )

    def set_sdxl_turbo(self):
        self.sd_dict.update(
            {
                "model": str(self.model_base_dir / "sd_xl_turbo_1.0_fp16.safetensors"),
                "vae": "",
                "height": 1024,
                "width": 1024,
                "type": "f16",
                "steps": 4,
            }
        )

    def set_sd_turbo(self):
        self.sd_dict.update(
            {
                "model": str(self.model_base_dir / "sd_turbo.safetensors"),
                "vae": "",
                "height": 512,
                "width": 512,
                "type": "f32",
                "steps": 4,
            }
        )

    def set_sd15(self):
        self.sd_dict.update(
            {
                "model": str(self.model_base_dir / "v1-5-pruned-emaonly.safetensors"),
                "vae": "",
                "height": 512,
                "width": 512,
                "type": "f32",
            }
        )

    def set_sd14(self):
        self.sd_dict.update(
            {
                "model": str(self.model_base_dir / "sd-v1-4.ckpt"),
                "vae": "",
                "height": 512,
                "width": 512,
                "type": "f32",
            }
        )

    def set_sd15_lcm(self):
        self.set_sd15()
        self.sd_dict.update({"steps": 4, "cfg-scale": 1})

    def generate(self, prompt: str, output: str):
        start_time = time.time()
        output_path = self.output_dir / output

        if output_path.exists():
            output_path = self.output_dir / output.replace(".png", "_1.png")

        self.sd_dict.update({"prompt": prompt, "output": str(output_path)})
        self._create_cmd()

        if not output_path.exists():
            logger.warning("Generation failed!")
            return

        img = Image.open(output_path)
        logger.info("Generation time: %.2f sec", time.time() - start_time)

        return img

    def _create_cmd(self):
        cmd = [self.sd_cmd]
        for k, v in self.sd_dict.items():
            cmd.append(f"--{k}")
            cmd.append(str(v))

        status = subprocess.run(cmd, check=False, text=True)
        return status.stdout
