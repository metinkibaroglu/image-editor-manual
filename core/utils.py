"""
Utility functions for image processing operations.
Contains low-level operations like convolution and kernel generation.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided


def manual_convolve(image_channel, kernel):
    """
    Performs 2D convolution (cross-correlation) manually on a single image channel using stride tricks.
    
    Args:
        image_channel (np.ndarray): 2D array representing a single image channel
        kernel (np.ndarray): 2D convolution kernel
        
    Returns:
        np.ndarray: Convolved image channel
    """
    k_h, k_w = kernel.shape
    img_h, img_w = image_channel.shape

    # Padding sizes
    pad_h = k_h // 2
    pad_w = k_w // 2

    # Pad the image channel
    padded_image = np.pad(
        image_channel, 
        ((pad_h, pad_h), (pad_w, pad_w)), 
        mode='constant', 
        constant_values=0
    )

    # Use as_strided to get view of all sliding windows
    strides = padded_image.strides
    view_shape = (img_h, img_w, k_h, k_w)
    view_strides = (strides[0], strides[1], strides[0], strides[1])

    # Create the strided view
    window_view = as_strided(padded_image, shape=view_shape, strides=view_strides)

    # Perform convolution using Einstein summation
    output = np.einsum('ijkl,kl->ij', window_view, kernel)
    return output


def create_gaussian_kernel(sigma, size=None):
    """
    Creates a 2D Gaussian kernel for blurring operations.
    
    Args:
        sigma (float): Standard deviation for the Gaussian distribution
        size (int, optional): Size of the kernel. If None, calculated automatically
        
    Returns:
        np.ndarray: 2D Gaussian kernel normalized to sum to 1
    """
    if size is None:
        # Determine size based on sigma
        size = int(2 * np.ceil(2 * sigma) + 1)
        if size % 2 == 0:
            size += 1 
    
    # Ensure minimum size and odd dimensions
    size = max(3, size)
    if size % 2 == 0: 
        size += 1

    kernel = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    variance = sigma ** 2

    # Avoid division by zero
    if variance == 0: 
        variance = 1e-6

    constant = 1.0 / (2 * np.pi * variance)

    # Generate Gaussian values
    for y in range(size):
        for x in range(size):
            dy = y - center
            dx = x - center
            exponent = -(dx**2 + dy**2) / (2 * variance)
            kernel[y, x] = constant * np.exp(exponent)

    # Normalize the kernel so it sums to 1
    return kernel / np.sum(kernel)


def create_sobel_kernels():
    """
    Creates Sobel kernels for edge detection.
    
    Returns:
        tuple: (sobel_x, sobel_y) kernels as numpy arrays
    """
    sobel_x = np.array([
        [-1, 0, 1], 
        [-2, 0, 2], 
        [-1, 0, 1]
    ], dtype=np.float64)
    
    sobel_y = np.array([
        [-1, -2, -1], 
        [0, 0, 0], 
        [1, 2, 1]
    ], dtype=np.float64)
    
    return sobel_x, sobel_y


def ensure_rgb_mode(image):
    """
    Ensures an image is in RGB mode.
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        PIL.Image: Image in RGB mode
    """
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image


def clip_to_uint8(array):
    """
    Clips array values to valid uint8 range and converts to uint8.
    
    Args:
        array (np.ndarray): Input array
        
    Returns:
        np.ndarray: Array with values clipped to [0, 255] and converted to uint8
    """
    return np.clip(array, 0, 255).astype(np.uint8) 