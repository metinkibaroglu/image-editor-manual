"""
Image filter functions for applying various visual effects.
"""

import numpy as np
from PIL import Image
from .utils import ensure_rgb_mode, clip_to_uint8, manual_convolve, create_gaussian_kernel, create_sobel_kernels
from .constants import SEPIA_MATRIX, LUMINOSITY_WEIGHTS


def apply_grayscale(image):
    """
    Applies grayscale filter using the luminosity method.
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        PIL.Image: Grayscale image
    """
    print("Applying manual grayscale...")
    image = ensure_rgb_mode(image)
    img_array = np.array(image, dtype=np.float32)

    # Calculate grayscale using luminosity weights
    gray_val = (LUMINOSITY_WEIGHTS['red'] * img_array[:, :, 0] +
                LUMINOSITY_WEIGHTS['green'] * img_array[:, :, 1] +
                LUMINOSITY_WEIGHTS['blue'] * img_array[:, :, 2])

    # Create 3-channel grayscale image
    gray_array = np.stack([gray_val] * 3, axis=-1)
    gray_array = clip_to_uint8(gray_array)
    
    return Image.fromarray(gray_array)


def apply_sepia(image):
    """
    Applies sepia tone filter using matrix transformation.
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        PIL.Image: Sepia-toned image
    """
    print("Applying manual sepia...")
    image = ensure_rgb_mode(image)
    img_array = np.array(image, dtype=np.float32)
    
    # Convert sepia matrix to numpy array
    sepia_matrix = np.array(SEPIA_MATRIX)

    # Reshape pixels for matrix multiplication
    pixels = img_array.reshape(-1, 3)
    transformed_pixels = pixels @ sepia_matrix.T
    sepia_array = transformed_pixels.reshape(img_array.shape)
    
    # Clip values and convert back to uint8
    sepia_array = clip_to_uint8(sepia_array)
    return Image.fromarray(sepia_array)


def apply_negative(image):
    """
    Applies negative (color inversion) filter.
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        PIL.Image: Negative image
    """
    print("Applying manual negative...")
    image = ensure_rgb_mode(image)
    img_array = np.array(image)
    
    # Invert colors by subtracting from 255
    negative_array = 255 - img_array
    negative_array = negative_array.astype(np.uint8)
    
    return Image.fromarray(negative_array)


def apply_blur(image, intensity):
    """
    Applies Gaussian blur using manual convolution.
    
    Args:
        image (PIL.Image): Input image
        intensity (float): Blur intensity (sigma for Gaussian kernel)
        
    Returns:
        PIL.Image: Blurred image
    """
    if intensity <= 0:
        return image
        
    print(f"Applying manual blur with intensity {intensity:.2f}...")
    image = ensure_rgb_mode(image)
    img_array = np.array(image, dtype=np.float64)
    blurred_array = np.zeros_like(img_array)

    # Create Gaussian kernel
    kernel = create_gaussian_kernel(intensity)
    print(f"Generated {kernel.shape[0]}x{kernel.shape[1]} Gaussian kernel for sigma={intensity:.2f}")

    # Apply convolution to each color channel
    for i in range(3):
        blurred_array[:, :, i] = manual_convolve(img_array[:, :, i], kernel)

    # Clip and convert back to uint8
    blurred_array = clip_to_uint8(blurred_array)
    return Image.fromarray(blurred_array)


def apply_sharpen(image, intensity):
    """
    Applies sharpening filter using manual unsharp masking.
    
    Args:
        image (PIL.Image): Input image
        intensity (float): Sharpening intensity
        
    Returns:
        PIL.Image: Sharpened image
    """
    if intensity <= 0:
        return image
        
    print(f"Applying manual sharpen with intensity {intensity:.2f}...")
    image = ensure_rgb_mode(image)
    img_array = np.array(image, dtype=np.float64)

    # Create a blurred version for unsharp masking
    sigma_unsharp = 1.0
    blur_kernel = create_gaussian_kernel(sigma_unsharp)
    blurred_component = np.zeros_like(img_array)
    
    for i in range(3):
        blurred_component[:, :, i] = manual_convolve(img_array[:, :, i], blur_kernel)

    # Calculate the detail (original - blurred)
    detail = img_array - blurred_component

    # Add scaled detail back: sharpened = original + intensity * detail
    sharpened_array = img_array + intensity * detail

    # Clip and convert back to uint8
    sharpened_array = clip_to_uint8(sharpened_array)
    return Image.fromarray(sharpened_array)


def apply_edge_detection(image):
    """
    Applies Sobel edge detection using manual convolution.
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        PIL.Image: Edge-detected image (grayscale)
    """
    print("Applying manual edge detection...")
    
    # Convert to grayscale for edge detection
    grayscale_image = image.convert('L')
    img_array = np.array(grayscale_image, dtype=np.float64)

    # Get Sobel operators
    sobel_x, sobel_y = create_sobel_kernels()

    # Apply manual convolution
    print("Applying Sobel X kernel...")
    grad_x = manual_convolve(img_array, sobel_x)
    print("Applying Sobel Y kernel...")
    grad_y = manual_convolve(img_array, sobel_y)

    # Calculate gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize magnitude to 0-255
    max_magnitude = np.max(magnitude)
    if max_magnitude > 0:
        magnitude *= 255.0 / max_magnitude
    else:
        magnitude = np.zeros_like(magnitude)

    # Convert back to uint8 image
    edge_array = magnitude.astype(np.uint8)
    return Image.fromarray(edge_array, mode='L') 