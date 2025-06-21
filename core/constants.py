"""
Constants and configuration values for the Image Editor application.
"""

# Application constants
APP_TITLE = "Image Editor"
APP_GEOMETRY = "1920x1080"
CONTROL_FRAME_WIDTH = 300

# Default values for image processing
DEFAULT_BRIGHTNESS = 1.0
DEFAULT_CONTRAST = 1.0
DEFAULT_SATURATION = 1.0
DEFAULT_BLUR_SIGMA = 1.0
DEFAULT_SHARPEN_INTENSITY = 1.0

# Slider ranges
BRIGHTNESS_RANGE = (0.0, 2.0)
CONTRAST_RANGE = (0.0, 2.0)
SATURATION_RANGE = (0.0, 2.0)
BLUR_RANGE = (0, 10)
SHARPEN_RANGE = (0, 5)

# Number of steps for sliders
SLIDER_STEPS = 100
BLUR_STEPS = 50
SHARPEN_STEPS = 50

# Sepia transformation matrix
SEPIA_MATRIX = [
    [0.393, 0.769, 0.189],
    [0.349, 0.686, 0.168],
    [0.272, 0.534, 0.131]
]

# Luminosity weights for grayscale conversion
LUMINOSITY_WEIGHTS = {
    'red': 0.299,
    'green': 0.587,
    'blue': 0.114
}

# Sobel operators for edge detection
SOBEL_X = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]

SOBEL_Y = [
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
]

# File dialog settings
IMAGE_FILETYPES = [
    ("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif"),
    ("All Files", "*.*"),
]

SAVE_FILETYPES = [
    ("PNG Image", "*.png"),
    ("JPEG Image", "*.jpg"),
    ("BMP Image", "*.bmp"),
]

# UI Color theme
DEFAULT_COLOR_THEME = "blue"
APPEARANCE_MODE = "System" 