# Image Editor - Manual Kernels & CustomTkinter GUI

A user-friendly desktop Image Editor application built with Python, `customtkinter`, and `Pillow`. This tool allows for various image adjustments and filter applications. A key feature is the manual implementation of convolution kernels for filters like blur, sharpen, and edge detection, offering a look into the underlying mechanics of image processing.

![Overall Application GUI](images/interface.png)

## Features

*   **Image Loading & Saving:** Load images from your computer and save them in various formats (PNG, JPG, BMP).
*   **Core Adjustments:** Sliders for real-time Brightness, Contrast, and Saturation adjustments.
*   **Transformations:** Easily Rotate (left/right) and Flip (horizontal/vertical) your images.
*   **Advanced Filters:** Apply a range of filters, including:
    *   Grayscale, Sepia, and Negative (Invert).
    *   **Manual Kernel Filters:** Gaussian Blur, Sharpen, and Edge Detection, all implemented from scratch to showcase the underlying algorithms.
*   **Interactive Cropping:** A simple tool to select and crop a specific region of your image.
*   **Reset Functionality:** Instantly revert all changes to the originally loaded image.

## Technologies Used

*   **Python 3.7+**
*   **CustomTkinter:** For the modern and responsive user interface.
*   **Pillow:** For robust image loading, saving, and manipulation.
*   **NumPy:** For high-performance numerical operations and manual filter implementations.

## Setup

1.  **Ensure you have Python 3.7 or newer installed.**
2.  **Clone the repository:**
    ```bash
    git clone https://github.com/metinkibaroglu/image-editor-manual.git
    cd image-editor-manual
    ```
3.  **Create and activate a virtual environment (Recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```
4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once the setup is complete, run the application from the root directory of the project:
```bash
python main.py
```
The main window will appear, allowing you to load an image and begin editing.

## How It Works: A Look Under the Hood

This project was intentionally built to demonstrate core image processing concepts.

*   **Package Structure:** The code is organized into a clean and maintainable package structure (`core`), separating UI, image operations, and utilities. This makes the project easy to understand, test, and extend.

*   **Manual Implementations:** Instead of relying solely on one-line library functions, key features are built from scratch:
    1.  **Convolution Engine:** Filters like Blur and Sharpen use a manually implemented 2D convolution function. This involves generating a specific kernel (e.g., a Gaussian kernel) and sliding it across the image's pixel data to compute the new, filtered values.
    2.  **Pixel-Level Adjustments:** Brightness, contrast, and saturation are adjusted by directly manipulating the image's pixel values in a NumPy array, providing insight into the color theory behind these enhancements.

This approach makes the project a great learning tool for those interested in the fundamentals of digital image processing.

## Available Operations (via GUI)

*   **Load Image:** Opens a dialog to select an image.
*   **Save As...:** Opens a dialog to save the currently displayed image.
*   **Reset Image:** Reverts all changes back to the original image.
*   **Adjustments (Sliders):** Control Brightness, Contrast, and Saturation.
*   **Transformations (Buttons):** Rotate Left/Right and Flip Horizontal/Vertical.
*   **Filters (Buttons & Sliders):**
    *   Apply Grayscale, Sepia, Negative, and Edge Detection filters.
    *   Use sliders to control the intensity of the manual Blur and Sharpen filters.
*   **Crop (Button):** Enter a mode to interactively draw a crop rectangle on the image.

## To-Do / Future Enhancements

*   [ ] Add more filter types (e.g., median filter, custom convolution kernel input).
*   [ ] Implement undo/redo functionality.
*   [ ] Display image metadata (EXIF).
*   [ ] Performance optimizations for very large images.
