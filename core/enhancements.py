"""
Image enhancement functions for adjusting brightness, contrast, and saturation.
"""

import numpy as np
from PIL import Image
from .utils import ensure_rgb_mode, clip_to_uint8
from .constants import LUMINOSITY_WEIGHTS


def apply_brightness(image, factor):
    """
    Adjusts image brightness manually using numpy.
    
    Args:
        image (PIL.Image): Input image
        factor (float): Brightness factor (1.0 = no change, >1.0 = brighter, <1.0 = darker)
        
    Returns:
        PIL.Image: Brightness-adjusted image
    """
    if factor == 1.0:
        return image  # No change needed
    
    print(f"Applying manual brightness: {factor:.2f}")
    image = ensure_rgb_mode(image)
    img_array = np.array(image, dtype=np.float32)

    # Multiply pixel values by the brightness factor
    adjusted_array = img_array * factor

    # Clip values and convert back to uint8
    adjusted_array = clip_to_uint8(adjusted_array)
    return Image.fromarray(adjusted_array)


def apply_contrast(image, factor):
    """
    Adjusts image contrast manually using numpy.
    
    Args:
        image (PIL.Image): Input image
        factor (float): Contrast factor (1.0 = no change, >1.0 = more contrast, <1.0 = less contrast)
        
    Returns:
        PIL.Image: Contrast-adjusted image
    """
    if factor == 1.0:
        return image  # No change needed
    
    print(f"Applying manual contrast: {factor:.2f}")
    image = ensure_rgb_mode(image)
    img_array = np.array(image, dtype=np.float32)

    # Calculate mean value for contrast adjustment
    mean_val = np.mean(img_array)

    # Adjust contrast around the mean
    adjusted_array = mean_val + factor * (img_array - mean_val)

    # Clip values and convert back to uint8
    adjusted_array = clip_to_uint8(adjusted_array)
    return Image.fromarray(adjusted_array)


def apply_saturation(image, factor):
    """
    Adjusts image saturation manually using numpy.
    
    Args:
        image (PIL.Image): Input image
        factor (float): Saturation factor (1.0 = no change, >1.0 = more saturated, <1.0 = less saturated)
        
    Returns:
        PIL.Image: Saturation-adjusted image
    """
    if factor == 1.0:
        return image  # No change needed
    
    print(f"Applying manual saturation: {factor:.2f}")
    image = ensure_rgb_mode(image)
    img_array = np.array(image, dtype=np.float32)

    # Calculate grayscale equivalent using luminosity weights
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    gray_lum = (LUMINOSITY_WEIGHTS['red'] * r + 
                LUMINOSITY_WEIGHTS['green'] * g + 
                LUMINOSITY_WEIGHTS['blue'] * b)
    gray_lum_3c = gray_lum[:, :, np.newaxis]

    # Adjust saturation by blending with grayscale
    adjusted_array = gray_lum_3c + factor * (img_array - gray_lum_3c)

    # Clip values and convert back to uint8
    adjusted_array = clip_to_uint8(adjusted_array)
    return Image.fromarray(adjusted_array) 