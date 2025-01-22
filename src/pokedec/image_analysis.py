from typing import Dict, Union

import numpy as np
from PIL import Image


def calculate_image_characteristics(filepath: Union[str, np.array], rgb: bool = True) -> Dict[str, float]:
    """Calculate image characteristics."""
    if isinstance(filepath, str):
        with Image.open(filepath) as img:
            img_np = np.array(img)
    else:
        img_np = filepath

    if rgb:
        # Calculate mean and standard deviation for each channel
        avg_brightness_green = np.mean(img_np[:, :, 1])
        contrast_green = np.std(img_np[:, :, 1])
        sharpness_green = np.mean(np.abs(np.gradient(img_np[:, :, 1].ravel())))
        avg_brightness_red = np.mean(img_np[:, :, 0])
        contrast_red = np.std(img_np[:, :, 0])
        sharpness_red = np.mean(np.abs(np.gradient(img_np[:, :, 0].ravel())))
        avg_brightness_blue = np.mean(img_np[:, :, 2])
        contrast_blue = np.std(img_np[:, :, 2])
        sharpness_blue = np.mean(np.abs(np.gradient(img_np[:, :, 2].ravel())))

        # Total image characteristics
        avg_brightness = np.mean(img_np)
        contrast = np.std(img_np)
        sharpness = np.mean(np.abs(np.gradient(img_np)))

        return {
            "avg_brightness_green": avg_brightness_green,
            "contrast_green": contrast_green,
            "sharpness_green": sharpness_green,
            "avg_brightness_red": avg_brightness_red,
            "contrast_red": contrast_red,
            "sharpness_red": sharpness_red,
            "avg_brightness_blue": avg_brightness_blue,
            "contrast_blue": contrast_blue,
            "sharpness_blue": sharpness_blue,
            "avg_brightness": avg_brightness,
            "contrast": contrast,
            "sharpness": sharpness,
        }
    else:
        avg_brightness = np.mean(img_np)
        contrast = np.std(img_np)
        sharpness = np.mean(np.abs(np.gradient(img_np.ravel())))

        return {
            "avg_brightness": avg_brightness,
            "contrast": contrast,
            "sharpness": sharpness,
        }
