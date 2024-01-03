# Sung-Cheol Kim, Copyright 2024. All rights reserved.

import subprocess

from PIL import Image


class SDHandler(object):
    def __init__(self, model_base_dir: str = "/home/skim/sd/models"):
        self.model_base_dir = model_base_dir
        self.model = ""
        self.vae = ""
        self.height = "1024"
        self.width = "1024"
        self.prompt = ""
        self.sd_cmd = "sd"
        self.output = ""

    def generate(self, prompt: str):
        self.prompt = prompt
        print(self._create_cmd())
        img = Image.open(self.output)
        return img

    def _create_cmd(self):
        cmd = [
            self.sd_cmd,
            "--model",
            self.model,
            "--vae",
            self.vae,
            "--output",
            self.output,
            "--height",
            self.height,
            "--width",
            self.width,
            "--prompt",
            self.prompt,
        ]
        status = subprocess.run(cmd, check=False, text=True)
        return status.stdout
