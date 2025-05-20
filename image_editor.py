import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from numpy.lib.stride_tricks import as_strided
import math

# Manual convolution
def _manual_convolve(image_channel, kernel):
    """Performs 2D convolution (cross-correlation) manually on a single image channel using stride tricks."""
    k_h, k_w = kernel.shape
    img_h, img_w = image_channel.shape

    # Padding sizes
    pad_h = k_h // 2
    pad_w = k_w // 2

    # Pad the image channel
    padded_image = np.pad(image_channel, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    # Use as_strided to get view of all sliding windows
    # Get the strides of the padded image array
    strides = padded_image.strides

    # Define the shape of the output view
    view_shape = (img_h, img_w, k_h, k_w)

    view_strides = (strides[0], strides[1], strides[0], strides[1])

    # Create the strided view
    window_view = as_strided(padded_image, shape=view_shape, strides=view_strides)

    output = np.einsum('ijkl,kl->ij', window_view, kernel)
    return output

# Kernel generation
def _create_gaussian_kernel(sigma, size=None):
    """Creates a 2D Gaussian kernel."""
    if size is None:
        # Determine size
        size = int(2 * np.ceil(2 * sigma) + 1)
        if size % 2 == 0:
            size += 1 
    size = max(3, size)
    if size % 2 == 0: size += 1

    kernel = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    variance = sigma ** 2

    if variance == 0: variance = 1e-6

    constant = 1.0 / (2 * np.pi * variance)

    for y in range(size):
        for x in range(size):
            dy = y - center
            dx = x - center
            exponent = -(dx**2 + dy**2) / (2 * variance)
            kernel[y, x] = constant * np.exp(exponent)

    # Normalize the kernel
    return kernel / np.sum(kernel)

# Enhancement Functions
def _apply_brightness(image, factor):
    """Adjusts image brightness manually using numpy."""
    if factor == 1.0: return image # No change
    print(f"Applying manual brightness: {factor:.2f}")
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image, dtype=np.float32)

    # Multiply pixel values by the factor
    adjusted_array = img_array * factor

    # Clip values and convert back to uint8
    adjusted_array = np.clip(adjusted_array, 0, 255).astype(np.uint8)
    return Image.fromarray(adjusted_array)

def _apply_contrast(image, factor):
    """Adjusts image contrast manually using numpy."""
    if factor == 1.0: return image
    print(f"Applying manual contrast: {factor:.2f}")
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image, dtype=np.float32)

    mean_val = np.mean(img_array)

    # Adjust contrast
    adjusted_array = mean_val + factor * (img_array - mean_val)

    # Clip values and convert back to uint8
    adjusted_array = np.clip(adjusted_array, 0, 255).astype(np.uint8)
    return Image.fromarray(adjusted_array)

def _apply_saturation(image, factor):
    """Adjusts image saturation manually using numpy."""
    if factor == 1.0: return image
    print(f"Applying manual saturation: {factor:.2f}")
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image, dtype=np.float32)

    # Calculate grayscale equivalent using luminosity weights
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    gray_lum = 0.299 * r + 0.587 * g + 0.114 * b
    gray_lum_3c = gray_lum[:, :, np.newaxis]

    # Adjust saturation
    adjusted_array = gray_lum_3c + factor * (img_array - gray_lum_3c)

    # Clip values and convert back to uint8
    adjusted_array = np.clip(adjusted_array, 0, 255).astype(np.uint8)
    return Image.fromarray(adjusted_array)

class ImageEditorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window setup
        self.title("Image Editor")
        self.geometry("1920x1080")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # Variables
        self.original_image = None
        self.processed_image = None
        self.display_image = None
        self.current_rotation = 0
        self.flip_lr_state = False
        self.flip_tb_state = False
        self.first_loaded_image = None # For keeping the first loaded image

        # Enhancement factors
        self.enhancement_factors = {
            "brightness": 1.0,
            "contrast": 1.0,
            "saturation": 1.0,
        }

        # Filter states and parameters
        self.filter_states = {
            "grayscale": False,
            "sepia": False,
            "negative": False,
            "blur": False,
            "sharpen": False,
            "edge_detect": False,
        }
        self.blur_intensity = 0.0 # Sigma for Gaussian blur
        self.sharpen_intensity = 0.0 # Factor for sharpening

        # Widget references
        self.brightness_slider = None
        self.contrast_slider = None
        self.saturation_slider = None
        # Slider references for blue and sharpen
        self.blur_slider = None
        self.sharpen_slider = None
        # Cropping state
        self.cropping_mode = False
        self.crop_start = None
        self.crop_end = None
        self.crop_rect_id = None

        # Toggle buttons for feedback
        self.filter_buttons = {}

        # Layout
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Control frame
        self.control_frame = ctk.CTkFrame(self, width=300)
        self.control_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.control_frame.pack_propagate(False) 

        # Image display label
        self.image_label = ctk.CTkLabel(self, text="Load an image to start", text_color="gray")
        self.image_label.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.create_widgets()
        # Initial placeholder setup if needed
        self._update_image_label_size()

    def _update_image_label_size(self):
        # Track the size of the image label for resizing
        self.image_label.update_idletasks()
        self.label_width = self.image_label.winfo_width()
        self.label_height = self.image_label.winfo_height()

    def _reset_state(self):
        print("Resetting state")
        self.current_rotation = 0
        self.flip_lr_state = False
        self.flip_tb_state = False
        self.enhancement_factors = {"brightness": 1.0, "contrast": 1.0, "saturation": 1.0}
        self.filter_states = {k: False for k in self.filter_states}
        self.blur_intensity = 0.0
        self.sharpen_intensity = 0.0

        # Reset sliders/controls visually
        if self.brightness_slider: self.brightness_slider.set(1.0)
        if self.contrast_slider: self.contrast_slider.set(1.0)
        if self.saturation_slider: self.saturation_slider.set(1.0)
        if self.blur_slider: self.blur_slider.set(0.0)
        if self.sharpen_slider: self.sharpen_slider.set(0.0)

        # Reset filter button appearances
        self._update_filter_button_appearances() 
        print("State reset complete.")

    def reset_image(self):
        """Resets all adjustments and filters to the original loaded image."""
        if self.first_loaded_image is None:
            print("Reset called, but no image loaded.")
            return
        print("Resetting image to very first loaded state.")
        self.original_image = self.first_loaded_image.copy()
        self._reset_state()
        self.apply_changes()

    def create_widgets(self):
        """Creates and places all widgets in the control frame."""

        # Top buttons
        top_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        top_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        top_frame.grid_columnconfigure((0, 1), weight=1)

        load_button = ctk.CTkButton(top_frame, text="Load Image", command=self.load_image)
        load_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        save_button = ctk.CTkButton(top_frame, text="Save As...", command=self.save_image)
        save_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        reset_button = ctk.CTkButton(self.control_frame, text="Reset Image", command=self.reset_image)
        reset_button.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        # Adjustments section
        adj_label = ctk.CTkLabel(self.control_frame, text="Adjustments", font=ctk.CTkFont(weight="bold"))
        adj_label.grid(row=2, column=0, padx=10, pady=(10, 2), sticky="w")

        adj_frame = ctk.CTkFrame(self.control_frame)
        adj_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        adj_frame.grid_columnconfigure(1, weight=1)

        # Brightness
        ctk.CTkLabel(adj_frame, text="Brightness:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.brightness_slider = ctk.CTkSlider(adj_frame, from_=0.0, to=2.0, number_of_steps=100, command=self._update_brightness)
        self.brightness_slider.set(1.0)
        self.brightness_slider.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        # Contrast
        ctk.CTkLabel(adj_frame, text="Contrast:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.contrast_slider = ctk.CTkSlider(adj_frame, from_=0.0, to=2.0, number_of_steps=100, command=self._update_contrast)
        self.contrast_slider.set(1.0)
        self.contrast_slider.grid(row=1, column=1, padx=5, pady=5, sticky='ew')

        # Saturation
        ctk.CTkLabel(adj_frame, text="Saturation:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.saturation_slider = ctk.CTkSlider(adj_frame, from_=0.0, to=2.0, number_of_steps=100, command=self._update_saturation)
        self.saturation_slider.set(1.0)
        self.saturation_slider.grid(row=2, column=1, padx=5, pady=5, sticky='ew')

        # Transform Section
        trans_label = ctk.CTkLabel(self.control_frame, text="Transform", font=ctk.CTkFont(weight="bold"))
        trans_label.grid(row=4, column=0, padx=10, pady=(10, 2), sticky="w")

        trans_frame = ctk.CTkFrame(self.control_frame)
        trans_frame.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
        trans_frame.grid_columnconfigure((0, 1), weight=1)

        rotate_left_btn = ctk.CTkButton(trans_frame, text="Rotate Left", command=self._rotate_left)
        rotate_left_btn.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        rotate_right_btn = ctk.CTkButton(trans_frame, text="Rotate Right", command=self._rotate_right)
        rotate_right_btn.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        flip_lr_btn = ctk.CTkButton(trans_frame, text="Flip Horizontal", command=self._flip_horizontal)
        flip_lr_btn.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        flip_tb_btn = ctk.CTkButton(trans_frame, text="Flip Vertical", command=self._flip_vertical)
        flip_tb_btn.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        # Filters section 
        self.filter_label = ctk.CTkLabel(self.control_frame, text="Filters", font=ctk.CTkFont(weight="bold"))
        self.filter_label.grid(row=5, column=0, padx=10, pady=(10, 2), sticky="w")

        manual_filter_frame = ctk.CTkFrame(self.control_frame)
        manual_filter_frame.grid(row=6, column=0, padx=10, pady=5, sticky="nsew") 
        manual_filter_frame.grid_columnconfigure((0, 1), weight=1)

        # Store button references
        self.filter_buttons['grayscale'] = ctk.CTkButton(manual_filter_frame, text="Grayscale", command=lambda: self._toggle_filter('grayscale'))
        self.filter_buttons['grayscale'].grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        self.filter_buttons['sepia'] = ctk.CTkButton(manual_filter_frame, text="Sepia", command=lambda: self._toggle_filter('sepia'))
        self.filter_buttons['sepia'].grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        self.filter_buttons['negative'] = ctk.CTkButton(manual_filter_frame, text="Negative", command=lambda: self._toggle_filter('negative'))
        self.filter_buttons['negative'].grid(row=1, column=0, padx=5, pady=5, sticky='ew')
        self.filter_buttons['edge_detect'] = ctk.CTkButton(manual_filter_frame, text="Edges", command=lambda: self._toggle_filter('edge_detect'))
        self.filter_buttons['edge_detect'].grid(row=1, column=1, padx=5, pady=5, sticky='ew')

        # Blur controls
        ctk.CTkLabel(manual_filter_frame, text="Blur:").grid(row=2, column=0, padx=5, pady=(10, 5), sticky='w')
        self.blur_slider = ctk.CTkSlider(manual_filter_frame, from_=0, to=10, number_of_steps=50, command=self._update_blur_intensity)
        self.blur_slider.set(0)
        self.blur_slider.grid(row=2, column=1, padx=5, pady=(10, 5), sticky='ew')

        # Sharpen controls
        ctk.CTkLabel(manual_filter_frame, text="Sharpen:").grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.sharpen_slider = ctk.CTkSlider(manual_filter_frame, from_=0, to=5, number_of_steps=50, command=self._update_sharpen_intensity)
        self.sharpen_slider.set(0)
        self.sharpen_slider.grid(row=3, column=1, padx=5, pady=5, sticky='ew')

        # Crop button
        crop_button = ctk.CTkButton(self.control_frame, text="Crop", command=self.enter_crop_mode)
        crop_button.grid(row=99, column=0, padx=10, pady=(20, 10), sticky="ew")

        self._get_button_colors()

    def _get_button_colors(self):
        """Store default and active button colors."""
        # Check if buttons exist before accessing configuration
        if not self.filter_buttons:
             self.after(100, self._get_button_colors)
             return
        try:
            self._default_button_fg_color = self.filter_buttons['grayscale'].cget("fg_color")
            theme_colors = ctk.ThemeManager.theme["color"]
            self._active_button_fg_color = theme_colors["button_hover"]
            print(f"Default button color: {self._default_button_fg_color}")
            print(f"Active button color: {self._active_button_fg_color}")
        except Exception as e:
            print(f"Error getting button colors: {e}")
            self._default_button_fg_color = ("#3B8ED0", "#1F6AA5") 
            self._active_button_fg_color = ("#36719F", "#144870") 

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("All Files", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            print(f"Loading image: {file_path}")
            img = Image.open(file_path).convert("RGB")
            if self.first_loaded_image is None:
                self.first_loaded_image = img.copy()
            self.original_image = img
            self._reset_state()
            self.apply_changes()
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            messagebox.showerror("Load Error", f"File not found:\n{file_path}")
            self._clear_image_display()
        except Exception as e:
            print(f"Error loading image: {e}")
            messagebox.showerror("Load Error", f"Could not load image file.\nError: {e}")
            self._clear_image_display()

    def _clear_image_display(self):
        """Clears the image variables and updates the label to its initial state."""
        self.original_image = None
        self.processed_image = None
        self.display_image = None
        self.ctk_image = None
        self.image_label.configure(text="Load an image to start", image=None)
        self.image_label.image = None

    def save_image(self):
        if not self.display_image:
            print("No image to save.")
            # TODO:
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Image As",
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg"),
                ("BMP Image", "*.bmp"),
            ],
        )
        if not file_path:
            return

        try:
            print(f"Saving image to: {file_path}")
            save_img = self.display_image
            if file_path.lower().endswith(('.jpg', '.jpeg')):
                if save_img.mode == 'RGBA':
                    print("Converting RGBA to RGB for JPEG save.")
                    save_img = save_img.convert('RGB')
                elif save_img.mode == 'P':
                    print("Converting Palette mode to RGB for JPEG save.")
                    save_img = save_img.convert('RGB')
                elif save_img.mode == 'L':
                     print("Converting Grayscale (L) to RGB for JPEG save.")
                     save_img = save_img.convert('RGB')

            save_img.save(file_path)
            print("Image saved successfully.")
        except Exception as e:
            print(f"Error saving image: {e}")
            messagebox.showerror("Save Error", f"Could not save image file.\nError: {e}")

    def apply_changes(self):
        """The main image processing pipeline."""
        if self.original_image is None:
            print("Apply changes called, but no original image loaded.")
            return

        print("--- Applying Changes Pipeline Start ---")
        image = self.original_image.copy()

        # Apply rotations/flips
        if self.current_rotation != 0:
            print(f"Rotating by {self.current_rotation} degrees")
            if self.current_rotation == 90:
                image = image.transpose(Image.Transpose.ROTATE_90)
            elif self.current_rotation == 180:
                image = image.transpose(Image.Transpose.ROTATE_180)
            elif self.current_rotation == 270:
                 image = image.transpose(Image.Transpose.ROTATE_270)

        if self.flip_lr_state:
            print("Flipping horizontally")
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        if self.flip_tb_state:
            print("Flipping vertically")
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        # Apply enhancements
        print(f"Applying Enhancements: B={self.enhancement_factors['brightness']:.2f}, C={self.enhancement_factors['contrast']:.2f}, S={self.enhancement_factors['saturation']:.2f}")
        if self.enhancement_factors['brightness'] != 1.0:
            image = _apply_brightness(image, self.enhancement_factors['brightness'])
        if self.enhancement_factors['contrast'] != 1.0:
            image = _apply_contrast(image, self.enhancement_factors['contrast'])
        if self.enhancement_factors['saturation'] != 1.0:
            image = _apply_saturation(image, self.enhancement_factors['saturation'])

        self.processed_image = image

        # Apply filters
        filtered_image = self.processed_image.copy()
        applied_filters = []

        # 1. Color filters
        if self.filter_states['grayscale']:
            filtered_image = _apply_grayscale(filtered_image)
            applied_filters.append('Grayscale')
        elif self.filter_states['sepia']:
            filtered_image = _apply_sepia(filtered_image)
            applied_filters.append('Sepia')

        if self.filter_states['negative']:
            filtered_image = _apply_negative(filtered_image)
            applied_filters.append('Negative')

        # 2. Spatial filters (Blur OR Sharpen)
        if self.filter_states['blur'] and self.blur_intensity > 0:
            filtered_image = _apply_blur(filtered_image, self.blur_intensity)
            applied_filters.append(f'Blur({self.blur_intensity:.1f})')
        elif self.filter_states['sharpen'] and self.sharpen_intensity > 0:
            # Ensure blur is not also applied
            filtered_image = _apply_sharpen(filtered_image, self.sharpen_intensity)
            applied_filters.append(f'Sharpen({self.sharpen_intensity:.1f})')

        # 3. Edge detection
        edge_active = self.filter_states['edge_detect']
        if edge_active:
            filtered_image = _apply_edge_detection(filtered_image)
            applied_filters.append('Edge Detect')
            if filtered_image.mode == 'L':
                print("Converting Edge Detection result back to RGB")
                filtered_image = filtered_image.convert('RGB')

        print(f"Applied Manual Filters: {', '.join(applied_filters) if applied_filters else 'None'}")
        self.display_image = filtered_image

        self.show_image() # Update the display
        print("--- Applying Changes Pipeline End ---")

    def show_image(self):
        """Updates the image displayed in the CTkLabel."""
        if self.display_image is None:
            self._clear_image_display()
            return

        print("Updating display...")
        self._update_image_label_size()

        img_w, img_h = self.display_image.size
        label_w, label_h = self.label_width, self.label_height

        if label_w <= 0 or label_h <= 0:
            print("Label size not determined yet, skipping display update.")
            self.after(100, self.show_image)
            return

        # Calculate aspect ratios
        img_aspect = img_w / img_h
        label_aspect = label_w / label_h

        # Determine new size
        if img_aspect > label_aspect:
            new_w = label_w
            new_h = int(new_w / img_aspect)
        else:
            new_h = label_h
            new_w = int(new_h * img_aspect)

        new_w = max(1, new_w)
        new_h = max(1, new_h)

        # Resize and create CTkImage
        try:
            resized_image = self.display_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.ctk_image = ImageTk.PhotoImage(resized_image)

            self.image_label.configure(text="", image=self.ctk_image)
            self.image_label.image = self.ctk_image
        except Exception as e:
            print(f"Error creating/displaying CTkImage: {e}")
            self.image_label.configure(text="Error displaying image", image=None)
            self.image_label.image = None

    # Slider callbacks
    def _update_brightness(self, value):
        self.enhancement_factors["brightness"] = float(value)
        self.apply_changes()

    def _update_contrast(self, value):
        self.enhancement_factors["contrast"] = float(value)
        self.apply_changes()

    def _update_saturation(self, value):
        self.enhancement_factors["saturation"] = float(value)
        self.apply_changes()

    def _update_blur_intensity(self, value):
        intensity = float(value)
        self.blur_intensity = intensity
        print(f"Blur slider updated: {intensity}")

        if intensity > 0:
            if not self.filter_states['blur']:
                self.filter_states['blur'] = True
            if self.filter_states['sharpen']:
                print("Deactivating Sharpen due to Blur adjustment")
                self.filter_states['sharpen'] = False
                self.sharpen_intensity = 0.0
                if self.sharpen_slider: self.sharpen_slider.set(0.0)
        else:
             if self.filter_states['blur']:
                 self.filter_states['blur'] = False

        self.apply_changes()

    def _update_sharpen_intensity(self, value):
        intensity = float(value)
        self.sharpen_intensity = intensity
        print(f"Sharpen slider updated: {intensity}")

        if intensity > 0:
            if not self.filter_states['sharpen']:
                self.filter_states['sharpen'] = True
            if self.filter_states['blur']:
                print("Deactivating Blur due to Sharpen adjustment")
                self.filter_states['blur'] = False
                self.blur_intensity = 0.0
                if self.blur_slider: self.blur_slider.set(0.0)
        else:
            if self.filter_states['sharpen']:
                self.filter_states['sharpen'] = False

        self.apply_changes()

    # Transform callbacks
    def _rotate_left(self):
        self.current_rotation = (self.current_rotation + 90) % 360
        self.apply_changes()

    def _rotate_right(self):
        self.current_rotation = (self.current_rotation - 90) % 360
        self.apply_changes()

    def _flip_horizontal(self):
        self.flip_lr_state = not self.flip_lr_state
        self.apply_changes()

    def _flip_vertical(self):
        self.flip_tb_state = not self.flip_tb_state
        self.apply_changes()

    # Filter toggle callbacks
    def _toggle_filter(self, filter_name):
        if self.original_image is None: return
        if not hasattr(self, '_default_button_fg_color'): return

        current_state = self.filter_states[filter_name]
        new_state = not current_state
        self.filter_states[filter_name] = new_state
        print(f"Toggling filter '{filter_name}' to {new_state}")

        # Handle blur/sharpen exclusivity
        if filter_name == 'blur':
            if new_state:
                if self.filter_states['sharpen']:
                    self.filter_states['sharpen'] = False
                    self.sharpen_intensity = 0.0
                    if self.sharpen_slider: self.sharpen_slider.set(0.0)
                if self.blur_intensity == 0.0 and self.blur_slider:
                    default_blur = 1.0
                    self.blur_intensity = default_blur
                    self.blur_slider.set(default_blur)
            else:
                self.blur_intensity = 0.0
                if self.blur_slider: self.blur_slider.set(0.0)

        elif filter_name == 'sharpen':
            if new_state:
                if self.filter_states['blur']:
                    self.filter_states['blur'] = False
                    self.blur_intensity = 0.0
                    if self.blur_slider: self.blur_slider.set(0.0)
                if self.sharpen_intensity == 0.0 and self.sharpen_slider:
                    default_sharpen = 1.0
                    self.sharpen_intensity = default_sharpen
                    self.sharpen_slider.set(default_sharpen)
            else: 
                self.sharpen_intensity = 0.0
                if self.sharpen_slider: self.sharpen_slider.set(0.0)

        # Update the appearance of all filter buttons based on the new states
        self._update_filter_button_appearances()

        # Apply all changes to the image
        self.apply_changes()

    def _update_filter_button_appearances(self):
        """Updates the visual state of filter buttons based on self.filter_states."""
        if not hasattr(self, '_default_button_fg_color'):
            print("Button colors not set yet, cannot update appearance.")
            return

        for name, button in self.filter_buttons.items():
            if self.filter_states[name]:
                button.configure(fg_color=self._active_button_fg_color)
            else:
                button.configure(fg_color=self._default_button_fg_color)

    def enter_crop_mode(self):
        if self.display_image is None:
            return
        self.cropping_mode = True
        self.crop_start = None
        self.crop_end = None
        self.crop_rect_id = None
        self.image_label.configure(cursor="crosshair")
        self.image_label.bind("<ButtonPress-1>", self._on_crop_press)
        self.image_label.bind("<B1-Motion>", self._on_crop_drag)
        self.image_label.bind("<ButtonRelease-1>", self._on_crop_release)
        print("Entered cropping mode. Drag to select area.")

    def exit_crop_mode(self):
        self.cropping_mode = False
        self.crop_start = None
        self.crop_end = None
        self.crop_rect_id = None
        self.image_label.configure(cursor="arrow")
        self.image_label.unbind("<ButtonPress-1>")
        self.image_label.unbind("<B1-Motion>")
        self.image_label.unbind("<ButtonRelease-1>")
        self.show_image() 
        print("Exited cropping mode.")

    def _on_crop_press(self, event):
        if not self.cropping_mode:
            return
        self.crop_start = (event.x, event.y)
        self.crop_end = (event.x, event.y)
        self._draw_crop_rectangle()

    def _on_crop_drag(self, event):
        if not self.cropping_mode or self.crop_start is None:
            return
        self.crop_end = (event.x, event.y)
        self._draw_crop_rectangle()

    def _on_crop_release(self, event):
        if not self.cropping_mode or self.crop_start is None:
            return
        self.crop_end = (event.x, event.y)
        self._draw_crop_rectangle(final=True)
        self._apply_crop()
        self.exit_crop_mode()

    def _draw_crop_rectangle(self, final=False):
        if self.display_image is None or self.crop_start is None or self.crop_end is None:
            return
        # Redraw image
        self._update_image_label_size()
        img_w, img_h = self.display_image.size
        label_w, label_h = self.label_width, self.label_height
        img_aspect = img_w / img_h
        label_aspect = label_w / label_h
        if img_aspect > label_aspect:
            new_w = label_w
            new_h = int(new_w / img_aspect)
        else:
            new_h = label_h
            new_w = int(new_h * img_aspect)
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        resized_image = self.display_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        # Draw rectangle overlay
        from PIL import ImageDraw
        overlay = resized_image.copy()
        draw = ImageDraw.Draw(overlay)
        x0, y0 = self.crop_start
        x1, y1 = self.crop_end
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        self.ctk_image = ImageTk.PhotoImage(overlay)
        self.image_label.configure(text="", image=self.ctk_image)
        self.image_label.image = self.ctk_image

    def _apply_crop(self):
        # Map crop rectangle from label coords to image coords
        if self.display_image is None or self.crop_start is None or self.crop_end is None:
            return
        img_w, img_h = self.display_image.size
        label_w, label_h = self.label_width, self.label_height
        img_aspect = img_w / img_h
        label_aspect = label_w / label_h
        if img_aspect > label_aspect:
            new_w = label_w
            new_h = int(new_w / img_aspect)
        else:
            new_h = label_h
            new_w = int(new_h * img_aspect)
        scale_x = img_w / new_w
        scale_y = img_h / new_h
        x0, y0 = self.crop_start
        x1, y1 = self.crop_end
        # Ensure coordinates are in correct order and within bounds
        left = int(max(0, min(x0, x1) * scale_x))
        upper = int(max(0, min(y0, y1) * scale_y))
        right = int(min(img_w, max(x0, x1) * scale_x))
        lower = int(min(img_h, max(y0, y1) * scale_y))
        if right - left < 2 or lower - upper < 2:
            print("Crop area too small, ignoring.")
            return
        print(f"Cropping image: left={left}, upper={upper}, right={right}, lower={lower}")

        img_array = np.array(self.display_image)
        cropped_array = img_array[upper:lower, left:right]
        cropped_img = Image.fromarray(cropped_array)
        self.display_image = cropped_img
        self.processed_image = cropped_img
        self.original_image = cropped_img
        self.show_image()

# Helper functions for filters
def _apply_grayscale(image):
    """Applies grayscale filter using the luminosity method."""
    print("Applying manual grayscale...")
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image, dtype=np.float32)

    gray_val = (0.299 * img_array[:, :, 0] +
                0.587 * img_array[:, :, 1] +
                0.114 * img_array[:, :, 2])

    gray_array = np.stack([gray_val]*3, axis=-1)

    gray_array = np.clip(gray_array, 0, 255).astype(np.uint8)
    return Image.fromarray(gray_array)

def _apply_sepia(image):
    """Applies sepia tone filter using matrix transformation."""
    print("Applying manual sepia...")
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image, dtype=np.float32)
    # Sepia matrix
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])

    pixels = img_array.reshape(-1, 3)
    transformed_pixels = pixels @ sepia_matrix.T
    sepia_array = transformed_pixels.reshape(img_array.shape)
    # Clip values and convert back to uint8
    sepia_array = np.clip(sepia_array, 0, 255).astype(np.uint8)
    return Image.fromarray(sepia_array)

def _apply_negative(image):
    """Applies negative (color inversion) filter."""
    print("Applying manual negative...")
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image)
    negative_array = 255 - img_array

    negative_array = negative_array.astype(np.uint8)
    return Image.fromarray(negative_array)

def _apply_blur(image, intensity):
    """Applies Gaussian blur using manual convolution."""
    print(f"Applying manual blur with intensity {intensity:.2f}...")
    if intensity <= 0:
        return image
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_array = np.array(image, dtype=np.float64)
    blurred_array = np.zeros_like(img_array)

    # Create kernel
    kernel = _create_gaussian_kernel(intensity)
    print(f"Generated {kernel.shape[0]}x{kernel.shape[1]} Gaussian kernel for sigma={intensity:.2f}")

    for i in range(3):
        blurred_array[:, :, i] = _manual_convolve(img_array[:, :, i], kernel)

    # Clip and convert back to uint8
    blurred_array = np.clip(blurred_array, 0, 255).astype(np.uint8)
    return Image.fromarray(blurred_array)

def _apply_sharpen(image, intensity):
    """Applies sharpening filter using manual unsharp masking."""
    print(f"Applying manual sharpen with intensity {intensity:.2f}...")
    if intensity <= 0:
        return image
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_array = np.array(image, dtype=np.float64)

    # 1. Create a blurred version
    sigma_unsharp = 1.0
    blur_kernel = _create_gaussian_kernel(sigma_unsharp)
    blurred_component = np.zeros_like(img_array)
    for i in range(3):
        blurred_component[:, :, i] = _manual_convolve(img_array[:, :, i], blur_kernel)

    # 2. Calculate the detail (original - blurred)
    detail = img_array - blurred_component

    # 3. Add scaled detail back: sharpened = original + intensity * detail
    sharpened_array = img_array + intensity * detail

    # 4. Clip and convert back to uint8
    sharpened_array = np.clip(sharpened_array, 0, 255).astype(np.uint8)
    return Image.fromarray(sharpened_array)

def _apply_edge_detection(image):
    """Applies Sobel edge detection using manual convolution."""
    print("Applying manual edge detection...")
    # 1. Convert to grayscale
    grayscale_image = image.convert('L')
    img_array = np.array(grayscale_image, dtype=np.float64)

    # 2. Define Sobel operators
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    # 3. Apply manual convolution
    print("Applying Sobel X kernel...")
    grad_x = _manual_convolve(img_array, sobel_x)
    print("Applying Sobel Y kernel...")
    grad_y = _manual_convolve(img_array, sobel_y)

    # 4. Calculate gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # 5. Normalize magnitude to 0-255
    max_magnitude = np.max(magnitude)
    if max_magnitude > 0:
        magnitude *= 255.0 / max_magnitude
    else:
        magnitude = np.zeros_like(magnitude)

    # 6. Convert back to uint8 image
    edge_array = magnitude.astype(np.uint8)
    return Image.fromarray(edge_array, mode='L')


if __name__ == "__main__":
    app = ImageEditorApp()
    app.mainloop()

# pyinstaller --noconfirm --onedir --noconsole "C:/path/to/python/file/python_file.py"