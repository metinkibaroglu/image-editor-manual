"""
Main GUI application for the Image Editor.
Contains the ImageEditorApp class with all interface logic and event handling.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np

from .constants import *
from .enhancements import apply_brightness, apply_contrast, apply_saturation
from .filters import apply_grayscale, apply_sepia, apply_negative, apply_blur, apply_sharpen, apply_edge_detection


class ImageEditorApp(ctk.CTk):
    """Main Image Editor Application GUI."""
    
    def __init__(self):
        super().__init__()
        self._setup_window()
        self._initialize_variables()
        self._setup_layout()
        self.create_widgets()
        self._update_image_label_size()

    def _setup_window(self):
        """Configure main window properties."""
        self.title(APP_TITLE)
        self.geometry(APP_GEOMETRY)
        ctk.set_appearance_mode(APPEARANCE_MODE)
        ctk.set_default_color_theme(DEFAULT_COLOR_THEME)

    def _initialize_variables(self):
        """Initialize all application state variables."""
        # Image variables
        self.original_image = None
        self.processed_image = None
        self.display_image = None
        self.first_loaded_image = None
        
        # Transformation state
        self.current_rotation = 0
        self.flip_lr_state = False
        self.flip_tb_state = False
        
        # Enhancement factors
        self.enhancement_factors = {
            "brightness": DEFAULT_BRIGHTNESS,
            "contrast": DEFAULT_CONTRAST,
            "saturation": DEFAULT_SATURATION,
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
        self.blur_intensity = 0.0
        self.sharpen_intensity = 0.0
        
        # Widget references
        self._initialize_widget_references()
        
        # Cropping state
        self.cropping_mode = False
        self.crop_start = None
        self.crop_end = None
        self.crop_rect_id = None

    def _initialize_widget_references(self):
        """Initialize widget reference variables."""
        self.brightness_slider = None
        self.contrast_slider = None
        self.saturation_slider = None
        self.blur_slider = None
        self.sharpen_slider = None
        self.filter_buttons = {}

    def _setup_layout(self):
        """Configure main application layout."""
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Control frame
        self.control_frame = ctk.CTkFrame(self, width=CONTROL_FRAME_WIDTH)
        self.control_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.control_frame.pack_propagate(False)

        # Image display label
        self.image_label = ctk.CTkLabel(self, text="Load an image to start", text_color="gray")
        self.image_label.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

    def create_widgets(self):
        """Create and place all widgets in the control frame."""
        self._create_file_controls()
        self._create_adjustment_controls()
        self._create_transform_controls()
        self._create_filter_controls()
        self._create_crop_control()
        self._get_button_colors()

    def _create_file_controls(self):
        """Create file operation buttons."""
        top_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        top_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        top_frame.grid_columnconfigure((0, 1), weight=1)

        load_button = ctk.CTkButton(top_frame, text="Load Image", command=self.load_image)
        load_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        save_button = ctk.CTkButton(top_frame, text="Save As...", command=self.save_image)
        save_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        reset_button = ctk.CTkButton(self.control_frame, text="Reset Image", command=self.reset_image)
        reset_button.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

    def _create_adjustment_controls(self):
        """Create image adjustment sliders."""
        adj_label = ctk.CTkLabel(self.control_frame, text="Adjustments", font=ctk.CTkFont(weight="bold"))
        adj_label.grid(row=2, column=0, padx=10, pady=(10, 2), sticky="w")

        adj_frame = ctk.CTkFrame(self.control_frame)
        adj_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        adj_frame.grid_columnconfigure(1, weight=1)

        # Brightness
        ctk.CTkLabel(adj_frame, text="Brightness:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.brightness_slider = ctk.CTkSlider(
            adj_frame, 
            from_=BRIGHTNESS_RANGE[0], 
            to=BRIGHTNESS_RANGE[1], 
            number_of_steps=SLIDER_STEPS, 
            command=self._update_brightness
        )
        self.brightness_slider.set(DEFAULT_BRIGHTNESS)
        self.brightness_slider.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        # Contrast
        ctk.CTkLabel(adj_frame, text="Contrast:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.contrast_slider = ctk.CTkSlider(
            adj_frame, 
            from_=CONTRAST_RANGE[0], 
            to=CONTRAST_RANGE[1], 
            number_of_steps=SLIDER_STEPS, 
            command=self._update_contrast
        )
        self.contrast_slider.set(DEFAULT_CONTRAST)
        self.contrast_slider.grid(row=1, column=1, padx=5, pady=5, sticky='ew')

        # Saturation
        ctk.CTkLabel(adj_frame, text="Saturation:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.saturation_slider = ctk.CTkSlider(
            adj_frame, 
            from_=SATURATION_RANGE[0], 
            to=SATURATION_RANGE[1], 
            number_of_steps=SLIDER_STEPS, 
            command=self._update_saturation
        )
        self.saturation_slider.set(DEFAULT_SATURATION)
        self.saturation_slider.grid(row=2, column=1, padx=5, pady=5, sticky='ew')

    def _create_transform_controls(self):
        """Create image transformation buttons."""
        trans_label = ctk.CTkLabel(self.control_frame, text="Transform", font=ctk.CTkFont(weight="bold"))
        trans_label.grid(row=4, column=0, padx=10, pady=(10, 2), sticky="w")

        trans_frame = ctk.CTkFrame(self.control_frame)
        trans_frame.grid(row=5, column=0, padx=10, pady=5, sticky="ew")
        trans_frame.grid_columnconfigure((0, 1), weight=1)

        rotate_left_btn = ctk.CTkButton(trans_frame, text="Rotate Left", command=self._rotate_left)
        rotate_left_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        rotate_right_btn = ctk.CTkButton(trans_frame, text="Rotate Right", command=self._rotate_right)
        rotate_right_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        flip_lr_btn = ctk.CTkButton(trans_frame, text="Flip Horizontal", command=self._flip_horizontal)
        flip_lr_btn.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        flip_tb_btn = ctk.CTkButton(trans_frame, text="Flip Vertical", command=self._flip_vertical)
        flip_tb_btn.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

    def _create_filter_controls(self):
        """Create filter buttons and controls."""
        filter_label = ctk.CTkLabel(self.control_frame, text="Filters", font=ctk.CTkFont(weight="bold"))
        filter_label.grid(row=6, column=0, padx=10, pady=(10, 2), sticky="w")

        filter_frame = ctk.CTkFrame(self.control_frame)
        filter_frame.grid(row=7, column=0, padx=10, pady=5, sticky="nsew")
        filter_frame.grid_columnconfigure((0, 1), weight=1)

        # Filter toggle buttons
        self.filter_buttons['grayscale'] = ctk.CTkButton(
            filter_frame, text="Grayscale", command=lambda: self._toggle_filter('grayscale')
        )
        self.filter_buttons['grayscale'].grid(row=0, column=0, padx=5, pady=5, sticky='ew')

        self.filter_buttons['sepia'] = ctk.CTkButton(
            filter_frame, text="Sepia", command=lambda: self._toggle_filter('sepia')
        )
        self.filter_buttons['sepia'].grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        self.filter_buttons['negative'] = ctk.CTkButton(
            filter_frame, text="Negative", command=lambda: self._toggle_filter('negative')
        )
        self.filter_buttons['negative'].grid(row=1, column=0, padx=5, pady=5, sticky='ew')

        self.filter_buttons['edge_detect'] = ctk.CTkButton(
            filter_frame, text="Edges", command=lambda: self._toggle_filter('edge_detect')
        )
        self.filter_buttons['edge_detect'].grid(row=1, column=1, padx=5, pady=5, sticky='ew')

        # Blur controls
        ctk.CTkLabel(filter_frame, text="Blur:").grid(row=2, column=0, padx=5, pady=(10, 5), sticky='w')
        self.blur_slider = ctk.CTkSlider(
            filter_frame, 
            from_=BLUR_RANGE[0], 
            to=BLUR_RANGE[1], 
            number_of_steps=BLUR_STEPS, 
            command=self._update_blur_intensity
        )
        self.blur_slider.set(0)
        self.blur_slider.grid(row=2, column=1, padx=5, pady=(10, 5), sticky='ew')

        # Sharpen controls
        ctk.CTkLabel(filter_frame, text="Sharpen:").grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.sharpen_slider = ctk.CTkSlider(
            filter_frame, 
            from_=SHARPEN_RANGE[0], 
            to=SHARPEN_RANGE[1], 
            number_of_steps=SHARPEN_STEPS, 
            command=self._update_sharpen_intensity
        )
        self.sharpen_slider.set(0)
        self.sharpen_slider.grid(row=3, column=1, padx=5, pady=5, sticky='ew')

    def _create_crop_control(self):
        """Create crop button."""
        crop_button = ctk.CTkButton(self.control_frame, text="Crop", command=self.enter_crop_mode)
        crop_button.grid(row=99, column=0, padx=10, pady=(20, 10), sticky="ew")

    def _get_button_colors(self):
        """Store default and active button colors for filter toggle feedback."""
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

    # Event handlers and core functionality methods will continue in the next part...
    
    def _update_image_label_size(self):
        """Track the size of the image label for resizing."""
        self.image_label.update_idletasks()
        self.label_width = self.image_label.winfo_width()
        self.label_height = self.image_label.winfo_height()

    def _reset_state(self):
        """Reset all image processing state to defaults."""
        print("Resetting state")
        self.current_rotation = 0
        self.flip_lr_state = False
        self.flip_tb_state = False
        self.enhancement_factors = {
            "brightness": DEFAULT_BRIGHTNESS, 
            "contrast": DEFAULT_CONTRAST, 
            "saturation": DEFAULT_SATURATION
        }
        self.filter_states = {k: False for k in self.filter_states}
        self.blur_intensity = 0.0
        self.sharpen_intensity = 0.0

        # Reset sliders/controls visually
        if self.brightness_slider: 
            self.brightness_slider.set(DEFAULT_BRIGHTNESS)
        if self.contrast_slider: 
            self.contrast_slider.set(DEFAULT_CONTRAST)
        if self.saturation_slider: 
            self.saturation_slider.set(DEFAULT_SATURATION)
        if self.blur_slider: 
            self.blur_slider.set(0.0)
        if self.sharpen_slider: 
            self.sharpen_slider.set(0.0)

        # Reset filter button appearances
        self._update_filter_button_appearances()
        print("State reset complete.")

    def reset_image(self):
        """Reset all adjustments and filters to the original loaded image."""
        if self.first_loaded_image is None:
            print("Reset called, but no image loaded.")
            return
        print("Resetting image to very first loaded state.")
        self.original_image = self.first_loaded_image.copy()
        self._reset_state()
        self.apply_changes()

    def load_image(self):
        """Load an image file using file dialog."""
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=IMAGE_FILETYPES,
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
        """Clear the image variables and update the label to its initial state."""
        self.original_image = None
        self.processed_image = None
        self.display_image = None
        self.ctk_image = None
        self.image_label.configure(text="Load an image to start", image=None)
        self.image_label.image = None

    def save_image(self):
        """Save the current image using file dialog."""
        if not self.display_image:
            print("No image to save.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Image As",
            defaultextension=".png",
            filetypes=SAVE_FILETYPES,
        )
        if not file_path:
            return

        try:
            print(f"Saving image to: {file_path}")
            save_img = self.display_image
            
            # Handle different image modes for JPEG saving
            if file_path.lower().endswith(('.jpg', '.jpeg')):
                if save_img.mode in ['RGBA', 'P', 'L']:
                    print(f"Converting {save_img.mode} to RGB for JPEG save.")
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
        print(f"Applying Enhancements: B={self.enhancement_factors['brightness']:.2f}, "
              f"C={self.enhancement_factors['contrast']:.2f}, S={self.enhancement_factors['saturation']:.2f}")
        
        if self.enhancement_factors['brightness'] != 1.0:
            image = apply_brightness(image, self.enhancement_factors['brightness'])
        if self.enhancement_factors['contrast'] != 1.0:
            image = apply_contrast(image, self.enhancement_factors['contrast'])
        if self.enhancement_factors['saturation'] != 1.0:
            image = apply_saturation(image, self.enhancement_factors['saturation'])

        self.processed_image = image

        # Apply filters
        filtered_image = self.processed_image.copy()
        applied_filters = []

        # Color filters
        if self.filter_states['grayscale']:
            filtered_image = apply_grayscale(filtered_image)
            applied_filters.append('Grayscale')
        elif self.filter_states['sepia']:
            filtered_image = apply_sepia(filtered_image)
            applied_filters.append('Sepia')

        if self.filter_states['negative']:
            filtered_image = apply_negative(filtered_image)
            applied_filters.append('Negative')

        # Spatial filters (Blur OR Sharpen)
        if self.filter_states['blur'] and self.blur_intensity > 0:
            filtered_image = apply_blur(filtered_image, self.blur_intensity)
            applied_filters.append(f'Blur({self.blur_intensity:.1f})')
        elif self.filter_states['sharpen'] and self.sharpen_intensity > 0:
            filtered_image = apply_sharpen(filtered_image, self.sharpen_intensity)
            applied_filters.append(f'Sharpen({self.sharpen_intensity:.1f})')

        # Edge detection
        if self.filter_states['edge_detect']:
            filtered_image = apply_edge_detection(filtered_image)
            applied_filters.append('Edge Detect')
            if filtered_image.mode == 'L':
                print("Converting Edge Detection result back to RGB")
                filtered_image = filtered_image.convert('RGB')

        print(f"Applied Manual Filters: {', '.join(applied_filters) if applied_filters else 'None'}")
        self.display_image = filtered_image

        self.show_image()
        print("--- Applying Changes Pipeline End ---")

    def show_image(self):
        """Update the image displayed in the CTkLabel."""
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

        # Calculate aspect ratios and resize
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

        try:
            resized_image = self.display_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.ctk_image = ImageTk.PhotoImage(resized_image)
            self.image_label.configure(text="", image=self.ctk_image)
            self.image_label.image = self.ctk_image
        except Exception as e:
            print(f"Error creating/displaying CTkImage: {e}")
            self.image_label.configure(text="Error displaying image", image=None)
            self.image_label.image = None

    # Slider callback methods
    def _update_brightness(self, value):
        """Update brightness factor and apply changes."""
        self.enhancement_factors["brightness"] = float(value)
        self.apply_changes()

    def _update_contrast(self, value):
        """Update contrast factor and apply changes."""
        self.enhancement_factors["contrast"] = float(value)
        self.apply_changes()

    def _update_saturation(self, value):
        """Update saturation factor and apply changes."""
        self.enhancement_factors["saturation"] = float(value)
        self.apply_changes()

    def _update_blur_intensity(self, value):
        """Update blur intensity and handle blur/sharpen exclusivity."""
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
                if self.sharpen_slider: 
                    self.sharpen_slider.set(0.0)
        else:
            if self.filter_states['blur']:
                self.filter_states['blur'] = False

        self.apply_changes()

    def _update_sharpen_intensity(self, value):
        """Update sharpen intensity and handle blur/sharpen exclusivity."""
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
                if self.blur_slider: 
                    self.blur_slider.set(0.0)
        else:
            if self.filter_states['sharpen']:
                self.filter_states['sharpen'] = False

        self.apply_changes()

    # Transform callback methods
    def _rotate_left(self):
        """Rotate image 90 degrees counterclockwise."""
        self.current_rotation = (self.current_rotation + 90) % 360
        self.apply_changes()

    def _rotate_right(self):
        """Rotate image 90 degrees clockwise."""
        self.current_rotation = (self.current_rotation - 90) % 360
        self.apply_changes()

    def _flip_horizontal(self):
        """Toggle horizontal flip state."""
        self.flip_lr_state = not self.flip_lr_state
        self.apply_changes()

    def _flip_vertical(self):
        """Toggle vertical flip state."""
        self.flip_tb_state = not self.flip_tb_state
        self.apply_changes()

    # Filter toggle methods
    def _toggle_filter(self, filter_name):
        """Toggle filter state and handle exclusivities."""
        if self.original_image is None: 
            return
        if not hasattr(self, '_default_button_fg_color'): 
            return

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
                    if self.sharpen_slider: 
                        self.sharpen_slider.set(0.0)
                if self.blur_intensity == 0.0 and self.blur_slider:
                    self.blur_intensity = DEFAULT_BLUR_SIGMA
                    self.blur_slider.set(DEFAULT_BLUR_SIGMA)
            else:
                self.blur_intensity = 0.0
                if self.blur_slider: 
                    self.blur_slider.set(0.0)

        elif filter_name == 'sharpen':
            if new_state:
                if self.filter_states['blur']:
                    self.filter_states['blur'] = False
                    self.blur_intensity = 0.0
                    if self.blur_slider: 
                        self.blur_slider.set(0.0)
                if self.sharpen_intensity == 0.0 and self.sharpen_slider:
                    self.sharpen_intensity = DEFAULT_SHARPEN_INTENSITY
                    self.sharpen_slider.set(DEFAULT_SHARPEN_INTENSITY)
            else:
                self.sharpen_intensity = 0.0
                if self.sharpen_slider: 
                    self.sharpen_slider.set(0.0)

        self._update_filter_button_appearances()
        self.apply_changes()

    def _update_filter_button_appearances(self):
        """Update the visual state of filter buttons based on self.filter_states."""
        if not hasattr(self, '_default_button_fg_color'):
            print("Button colors not set yet, cannot update appearance.")
            return

        for name, button in self.filter_buttons.items():
            if self.filter_states[name]:
                button.configure(fg_color=self._active_button_fg_color)
            else:
                button.configure(fg_color=self._default_button_fg_color)

    # Cropping methods
    def enter_crop_mode(self):
        """Enter interactive cropping mode."""
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
        """Exit cropping mode and restore normal interaction."""
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
        """Handle mouse press for crop selection."""
        if not self.cropping_mode:
            return
        self.crop_start = (event.x, event.y)
        self.crop_end = (event.x, event.y)
        self._draw_crop_rectangle()

    def _on_crop_drag(self, event):
        """Handle mouse drag for crop selection."""
        if not self.cropping_mode or self.crop_start is None:
            return
        self.crop_end = (event.x, event.y)
        self._draw_crop_rectangle()

    def _on_crop_release(self, event):
        """Handle mouse release to finalize crop selection."""
        if not self.cropping_mode or self.crop_start is None:
            return
        self.crop_end = (event.x, event.y)
        self._draw_crop_rectangle(final=True)
        self._apply_crop()
        self.exit_crop_mode()

    def _draw_crop_rectangle(self, final=False):
        """Draw crop selection rectangle overlay."""
        if (self.display_image is None or 
            self.crop_start is None or 
            self.crop_end is None):
            return

        # Calculate display dimensions
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
        
        # Create overlay with crop rectangle
        resized_image = self.display_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        overlay = resized_image.copy()
        draw = ImageDraw.Draw(overlay)
        
        x0, y0 = self.crop_start
        x1, y1 = self.crop_end
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        
        self.ctk_image = ImageTk.PhotoImage(overlay)
        self.image_label.configure(text="", image=self.ctk_image)
        self.image_label.image = self.ctk_image

    def _apply_crop(self):
        """Apply the crop selection to the image."""
        if (self.display_image is None or 
            self.crop_start is None or 
            self.crop_end is None):
            return

        # Calculate scaling from display to actual image coordinates
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
        
        # Convert to image coordinates and ensure proper bounds
        left = int(max(0, min(x0, x1) * scale_x))
        upper = int(max(0, min(y0, y1) * scale_y))
        right = int(min(img_w, max(x0, x1) * scale_x))
        lower = int(min(img_h, max(y0, y1) * scale_y))
        
        if right - left < 2 or lower - upper < 2:
            print("Crop area too small, ignoring.")
            return
            
        print(f"Cropping image: left={left}, upper={upper}, right={right}, lower={lower}")
        
        # Apply crop
        img_array = np.array(self.display_image)
        cropped_array = img_array[upper:lower, left:right]
        cropped_img = Image.fromarray(cropped_array)
        
        self.display_image = cropped_img
        self.processed_image = cropped_img
        self.original_image = cropped_img
        self.show_image() 