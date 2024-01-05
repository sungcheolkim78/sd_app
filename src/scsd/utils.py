# Sung-Cheol Kim, Copyright 2023. All rights reserved.

import logging
from typing import List

from diffusers.utils.pil_utils import make_image_grid
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LDMHandler")
logger.setLevel(logging.INFO)


def grid_images(images: List[Image.Image]):
    """
    Creates a grid of images based on the number of images provided.

    Args:
        images (List[Image.Image]): A list of PIL Image objects.

    Returns:
        Image.Image: The grid image.

    Raises:
        None
    """

    if len(images) == 1:
        return images[0]
    elif len(images) <= 4:
        return make_image_grid(images, rows=1, cols=len(images))
    else:
        return make_image_grid(images, rows=2, cols=len(images) // 2)
