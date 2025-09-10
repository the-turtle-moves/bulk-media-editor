import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
import os
import sys
import shutil
from PIL import Image, ImageTk, ImageDraw
import mediapipe as mp
from image_processor import process_images, resource_path, multiline_bbox, get_automatic_placement_coords, safe_print, replace_quotes, generate_captioned_image, wrap_text, get_video_frame, resize_and_crop
import random
import threading
import queue
import re

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.video_frame_cache = {}
        self.title("Image Captioner")
        self.geometry("1000x700")

        # --- Data ---
        self.image_list = [] # The single source of truth for image paths
        self.current_preview_image = None
        self.original_image_for_preview = None
        self.preview_image_path = None

        self.image_settings = {}  # path -> {'x': None, 'y': None, 'scale_x': 1.0, 'scale_y': 1.0, 'manual_placement': False, 'caption': ''}
        self.drag_info = {'start_x_offset': 0, 'start_y_offset': 0, 'is_dragging': False}
        self.resize_info = {'is_resizing': False, 'start_bbox': (0,0,0,0), 'start_scale_x': 1.0, 'start_scale_y': 1.0}
        self.caption_bbox_on_preview = None
        self.resize_handle_bbox = None
        self.updating_caption_box = False
        self.caption_update_timer = None # For debouncing caption input
        self.progress_queue = queue.Queue()

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        
        # --- Load Config ---
        try:
            # Look for config.json next to the executable or script
            base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)
            config_path = os.path.join(base_dir, 'config.json')
            
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            messagebox.showerror("Config Error", "config.json not found! Please make sure it's in the same directory as the application.")
            self.destroy()
            return
        except Exception as e:
            messagebox.showerror("Config Error", f"Failed to load or parse config.json: {e}")
            self.destroy()
            return

        # --- UI Elements ---
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Left Panel (File List)
        self.list_frame = tk.Frame(self.main_frame, width=200)
        self.list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.list_frame.pack_propagate(False)
        self.add_media_button = tk.Button(self.list_frame, text="Add Media", command=self.add_media)
        self.add_media_button.pack(pady=5, fill=tk.X)
        self.remove_button = tk.Button(self.list_frame, text="Remove Selected", command=self.remove_selected)
        self.remove_button.pack(pady=5, fill=tk.X)
        self.select_all_button = tk.Button(self.list_frame, text="Select All", command=self.select_all_images)
        self.select_all_button.pack(pady=5, fill=tk.X)
        self.listbox = tk.Listbox(self.list_frame, selectmode=tk.EXTENDED)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self.listbox.bind('<<ListboxSelect>>', self.on_file_select)

        # Info Labels
        self.info_frame = tk.Frame(self.list_frame)
        self.info_frame.pack(fill=tk.X, pady=5)
        self.total_images_var = tk.StringVar(value="Total: 0")
        self.total_images_label = tk.Label(self.info_frame, textvariable=self.total_images_var, anchor=tk.W)
        self.total_images_label.pack(fill=tk.X, padx=5)
        self.selected_images_var = tk.StringVar(value="Selected: 0")
        self.selected_images_label = tk.Label(self.info_frame, textvariable=self.selected_images_var, anchor=tk.W)
        self.selected_images_label.pack(fill=tk.X, padx=5)

        # Center Panel (Image Preview)
        self.preview_frame = tk.Frame(self.main_frame, bg='gray80')
        self.preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.preview_label = tk.Label(self.preview_frame, bg='gray80')
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        self.preview_label.bind("<ButtonPress-1>", self.on_mouse_press)
        self.preview_label.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.preview_label.bind("<B1-Motion>", self.on_mouse_motion)


        # Right Panel (Controls)
        self.control_frame = tk.Frame(self.main_frame, width=200)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self.control_frame.pack_propagate(False)

        # Sync Adjustments
        self.sync_frame = tk.LabelFrame(self.control_frame, text="Sync Adjustments")
        self.sync_frame.pack(pady=2, fill=tk.X)
        self.apply_to_all_var = tk.BooleanVar(value=True)
        self.apply_to_all_checkbox = tk.Checkbutton(self.sync_frame, text="Apply position/scale to all", variable=self.apply_to_all_var)
        self.apply_to_all_checkbox.pack(anchor=tk.W, padx=5, pady=2)
        self.sync_caption_var = tk.BooleanVar(value=True)
        self.sync_caption_checkbox = tk.Checkbutton(self.sync_frame, text="Use one caption for all", variable=self.sync_caption_var)
        self.sync_caption_checkbox.pack(anchor=tk.W, padx=5, pady=2)
        self.random_tilt_var = tk.BooleanVar(value=False)
        self.random_tilt_checkbox = tk.Checkbutton(self.sync_frame, text="Randomly tilt captions", variable=self.random_tilt_var, command=self.on_tilt_toggle)
        self.random_tilt_checkbox.pack(anchor=tk.W, padx=5, pady=2)
        self.font_outline_var = tk.BooleanVar(value=True)
        self.font_outline_checkbox = tk.Checkbutton(self.sync_frame, text="Enable font outline", variable=self.font_outline_var, command=lambda: self.display_image() if self.preview_image_path else None)
        self.font_outline_checkbox.pack(anchor=tk.W, padx=5, pady=2)
        self.recenter_button = tk.Button(self.sync_frame, text="Recenter All Captions", command=self.recenter_all_captions)
        self.recenter_button.pack(fill=tk.X, padx=5, pady=(2, 5))
        self.sync_caption_var.trace_add('write', self.on_sync_caption_toggle)

        # Grouping Options
        self.grouping_frame = tk.LabelFrame(self.control_frame, text="Grouping Options")
        self.grouping_frame.pack(pady=2, fill=tk.X, padx=2)
        self.group_size_label = tk.Label(self.grouping_frame, text="Number of groups (1 for all):")
        self.group_size_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.group_size_var = tk.StringVar(value="1")
        self.group_size_entry = tk.Entry(self.grouping_frame, textvariable=self.group_size_var, width=5)
        self.group_size_entry.pack(side=tk.LEFT, padx=5, pady=5)
        self.separate_folders_var = tk.BooleanVar(value=False)
        self.separate_folders_checkbox = tk.Checkbutton(self.grouping_frame, text="Save in separate folders", variable=self.separate_folders_var)
        self.separate_folders_checkbox.pack(side=tk.LEFT, padx=5, pady=5)

        # Outline Settings
        self.outline_frame = tk.LabelFrame(self.control_frame, text="Outline Settings")
        self.outline_frame.pack(pady=2, fill=tk.X)
        
        self.stroke_width_var = tk.IntVar(value=self.config.get('stroke_width', 4))
        
        self.outline_thickness_label = tk.Label(self.outline_frame, text="Thickness:")
        self.outline_thickness_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.outline_thickness_spinbox = tk.Spinbox(
            self.outline_frame, 
            from_=0, 
            to=100, 
            textvariable=self.stroke_width_var, 
            width=5
        )
        self.outline_thickness_spinbox.pack(side=tk.LEFT, padx=5, pady=5)
        self.stroke_width_var.trace_add('write', self.on_stroke_width_change)

        self.save_settings_button = tk.Button(self.outline_frame, text="Save", command=self.save_config)
        self.save_settings_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # Font Settings
        self.font_frame = tk.LabelFrame(self.control_frame, text="Font")
        self.font_frame.pack(pady=2, fill=tk.X)
        current_font = os.path.basename(self.config.get('font_path', '')) or 'Default'
        self.font_path_var = tk.StringVar(value=current_font)
        self.font_label = tk.Label(self.font_frame, textvariable=self.font_path_var, anchor=tk.W)
        self.font_label.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        self.font_button = tk.Button(self.font_frame, text="Choose...", command=self.choose_font)
        self.font_button.pack(side=tk.RIGHT, padx=5, pady=5)


        # Caption Input
        self.caption_frame = tk.LabelFrame(self.control_frame, text="Caption Text")
        self.caption_frame.pack(pady=2, fill=tk.X)
        self.caption_text_box = tk.Text(self.caption_frame, height=4)
        self.caption_text_box.pack(pady=5, padx=5, fill=tk.X)
        self.caption_text_box.bind('<<Modified>>', self.on_caption_change)

        self.overlay_image_path = None
        self.overlay_image_label_var = tk.StringVar(value="No overlay image selected")
        
        self.select_overlay_button = tk.Button(self.caption_frame, text="Select Overlay Image", command=self.select_overlay_image)
        self.select_overlay_button.pack(pady=5, padx=5, fill=tk.X)
        
        self.overlay_image_label = tk.Label(self.caption_frame, textvariable=self.overlay_image_label_var)
        self.overlay_image_label.pack(pady=5, padx=5, fill=tk.X)

        # Output Folder
        self.output_folder_frame = tk.LabelFrame(self.control_frame, text="Output Folder")
        self.output_folder_frame.pack(pady=2, fill=tk.X)
        self.output_folder_path = tk.StringVar(value=self.config.get('output_folder', 'captioned_images'))
        self.output_folder_entry = tk.Entry(self.output_folder_frame, textvariable=self.output_folder_path)
        self.output_folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.browse_button = tk.Button(self.output_folder_frame, text="...", command=self.select_output_folder)
        self.browse_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # Filename Options
        self.filename_frame = tk.LabelFrame(self.control_frame, text="Filename Options")
        self.filename_frame.pack(pady=2, fill=tk.X)
        self.filename_option = tk.StringVar(value="random") # Default to random
        self.random_name_radio = tk.Radiobutton(self.filename_frame, text="Random", variable=self.filename_option, value="random")
        self.random_name_radio.pack(anchor=tk.W, padx=5)
        self.sequential_name_radio = tk.Radiobutton(self.filename_frame, text="Sequential", variable=self.filename_option, value="sequential")
        self.sequential_name_radio.pack(anchor=tk.W, padx=5)

        # Resolution Settings
        self.resolution_frame = tk.LabelFrame(self.control_frame, text="Resize Options")
        self.resolution_frame.pack(pady=2, fill=tk.X)
        self.resize_enabled = tk.BooleanVar(value=False)
        self.resize_checkbox = tk.Checkbutton(self.resolution_frame, text="Resize images to custom resolution", variable=self.resize_enabled)
        self.resize_checkbox.pack(anchor=tk.W)
        self.resolution_entry_frame = tk.Frame(self.resolution_frame)
        self.resolution_entry_frame.pack(fill=tk.X)
        self.width_label = tk.Label(self.resolution_entry_frame, text="Width:")
        self.width_label.pack(side=tk.LEFT, padx=5)
        self.width_var = tk.StringVar(value="1080")
        self.width_entry = tk.Entry(self.resolution_entry_frame, textvariable=self.width_var, width=7)
        self.width_entry.pack(side=tk.LEFT)
        self.height_label = tk.Label(self.resolution_entry_frame, text="Height:")
        self.height_label.pack(side=tk.LEFT, padx=5)
        self.height_var = tk.StringVar(value="1920")
        self.height_entry = tk.Entry(self.resolution_entry_frame, textvariable=self.height_var, width=7)
        self.height_entry.pack(side=tk.LEFT)

        self.processing_buttons_frame = tk.Frame(self.control_frame)
        self.processing_buttons_frame.pack(pady=2, fill=tk.X)

        self.start_button = tk.Button(self.processing_buttons_frame, text="Process Images", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.process_video_button = tk.Button(self.processing_buttons_frame, text="Process Videos", command=self.start_video_processing)
        self.process_video_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

        self.progress_bar = ttk.Progressbar(self.control_frame, orient='horizontal', mode='determinate')

        # Bulk Preview button (renders thumbnails for selected images)
        self.bulk_preview_button = tk.Button(self.control_frame, text="Bulk Preview", command=self.start_bulk_preview)
        self.bulk_preview_button.pack(pady=(4, 0), fill=tk.X)

    def select_overlay_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Overlay Image",
            filetypes=(("Image Files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*"))
        )
        if file_path:
            self.overlay_image_path = file_path
            self.overlay_image_label_var.set(os.path.basename(file_path))
            self.caption_text_box.delete("1.0", tk.END)
            self.display_image()

    def show_error_popup(self, traceback_str):
        popup = tk.Toplevel(self)
        popup.title("Detailed Error Report")
        popup.geometry("700x500")
        text_area = tk.Text(popup, wrap=tk.WORD, font=("Courier New", 10))
        text_area.insert(tk.END, traceback_str)
        text_area.config(state=tk.DISABLED)
        
        scrollbar = tk.Scrollbar(text_area, command=text_area.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_area.config(yscrollcommand=scrollbar.set)
        
        text_area.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        close_button = tk.Button(popup, text="Close", command=popup.destroy)
        close_button.pack(pady=10)

    def on_mouse_press(self, event):
        if not self.preview_image_path: return

        if self.resize_handle_bbox and (self.resize_handle_bbox[0] <= event.x <= self.resize_handle_bbox[2] and self.resize_handle_bbox[1] <= event.y <= self.resize_handle_bbox[3]):
            self.resize_info['is_resizing'] = True
            settings = self.image_settings[self.preview_image_path]
            
            self.resize_info['start_bbox'] = self.caption_bbox_on_preview
            self.resize_info['start_scale_x'] = settings.get('scale_x', 1.0)
            self.resize_info['start_scale_y'] = settings.get('scale_y', 1.0)
            self.resize_info['start_mouse_x'] = event.x
            self.resize_info['start_mouse_y'] = event.y

        elif self.caption_bbox_on_preview and (self.caption_bbox_on_preview[0] <= event.x <= self.caption_bbox_on_preview[2] and self.caption_bbox_on_preview[1] <= event.y <= self.caption_bbox_on_preview[3]):
            self.drag_info['is_dragging'] = True
            self.drag_info['start_x_offset'] = event.x - self.caption_bbox_on_preview[0]
            self.drag_info['start_y_offset'] = event.y - self.caption_bbox_on_preview[1]

    def on_mouse_motion(self, event):
        if not self.preview_image_path or self.original_image_for_preview is None: return

        if self.resize_info['is_resizing']:
            settings = self.image_settings[self.preview_image_path]
            
            start_bbox = self.resize_info['start_bbox']
            start_w = start_bbox[2] - start_bbox[0]
            start_h = start_bbox[3] - start_bbox[1]

            if start_w <= 0 or start_h <= 0: return

            delta_x = event.x - self.resize_info['start_mouse_x']
            delta_y = event.y - self.resize_info['start_mouse_y']

            new_w = start_w + delta_x
            new_h = start_h + delta_y
            
            scale_factor_x = new_w / start_w if start_w > 0 else 1
            scale_factor_y = new_h / start_h if start_h > 0 else 1

            new_scale_x = max(0.1, self.resize_info['start_scale_x'] * scale_factor_x)
            new_scale_y = max(0.1, self.resize_info['start_scale_y'] * scale_factor_y)

            if self.apply_to_all_var.get():
                for path in self.image_list:
                    if path not in self.image_settings:
                        self.image_settings[path] = {'x': None, 'y': None, 'scale_x': 1.0, 'scale_y': 1.0, 'manual_placement': False, 'caption': ''}
                    self.image_settings[path]['scale_x'] = new_scale_x
                    self.image_settings[path]['scale_y'] = new_scale_y
            else:
                settings['scale_x'] = new_scale_x
                settings['scale_y'] = new_scale_y
            
            self.display_image()

        elif self.drag_info['is_dragging']:
            settings = self.image_settings[self.preview_image_path]
            
            img_w, img_h = self.original_image_for_preview.size
            panel_width = self.preview_frame.winfo_width()
            panel_height = self.preview_frame.winfo_height()
            ratio = min(panel_width / img_w, panel_height / img_h)
            
            preview_w, preview_h = int(img_w * ratio), int(img_h * ratio)
            offset_x = (panel_width - preview_w) / 2
            offset_y = (panel_height - preview_h) / 2

            new_preview_x = event.x - self.drag_info['start_x_offset']
            new_preview_y = event.y - self.drag_info['start_y_offset']

            new_x = (new_preview_x - offset_x) / ratio
            new_y = (new_preview_y - offset_y) / ratio

            if self.apply_to_all_var.get():
                for path in self.image_list:
                    if path not in self.image_settings:
                        self.image_settings[path] = {'x': None, 'y': None, 'scale_x': 1.0, 'scale_y': 1.0, 'manual_placement': False, 'caption': ''}
                    self.image_settings[path]['x'] = new_x
                    self.image_settings[path]['y'] = new_y
                    self.image_settings[path]['manual_placement'] = True
            else:
                settings['x'] = new_x
                settings['y'] = new_y
                settings['manual_placement'] = True
            
            self.display_image()

    def on_mouse_release(self, event):
        self.drag_info['is_dragging'] = False
        self.resize_info['is_resizing'] = False

    def choose_font(self):
        """Open a file dialog to choose a font file (.otf/.ttf). If canceled, keep current font."""
        file_path = filedialog.askopenfilename(
            title="Select Font File",
            filetypes=(("Font Files", "*.otf;*.ttf"), ("OTF", "*.otf"), ("TTF", "*.ttf"), ("All files", "*.*"))
        )
        if not file_path:
            return  # Default to current font if none selected

        try:
            # Update config and label
            self.config['font_path'] = file_path
            self.font_path_var.set(os.path.basename(file_path))

            # Re-wrap and refresh current preview to reflect new font metrics
            if self.preview_image_path and self.preview_image_path in self.image_settings:
                settings = self.image_settings[self.preview_image_path]
                caption_text = settings.get('caption', '')
                if isinstance(self.original_image_for_preview, Image.Image):
                    img_w, _ = self.original_image_for_preview.size
                else:
                    img_w, _ = self.original_image_for_preview.shape[1], self.original_image_for_preview.shape[0]
                wrapped = wrap_text(
                    caption_text,
                    resource_path(self.config['font_path']),
                    self.config['font_size'],
                    self.config['text_width_ratio'],
                    img_w
                )
                settings['wrapped_caption'] = wrapped
                self.display_image()
        except Exception as e:
            messagebox.showerror("Font Error", f"Failed to set font: {e}")

    def _update_listbox(self):
        current_selection_indices = self.listbox.curselection()
        
        self.listbox.delete(0, tk.END)
        for i, path in enumerate(self.image_list):
            display_text = f"{i+1}. {os.path.basename(path)}"
            self.listbox.insert(tk.END, display_text)

        # Restore selection
        for i in current_selection_indices:
            if i < self.listbox.size():
                self.listbox.select_set(i)
        
        self._update_counts()

    def _update_counts(self):
        total_count = len(self.image_list)
        selected_count = len(self.listbox.curselection())
        self.total_images_var.set(f"Total: {total_count}")
        self.selected_images_var.set(f"Selected: {selected_count}")

    

    def on_sync_caption_toggle(self, *args):
        is_synced = self.sync_caption_var.get()
        current_caption = self.caption_text_box.get("1.0", tk.END).strip()

        if not is_synced:
            # When switching from SYNCED to UNSYNCED, copy the current "global" caption to all individual images
            for path in self.image_list:
                if path in self.image_settings:
                    self.image_settings[path]['caption'] = current_caption
        else:
            # When switching from UNSYNCED to SYNCED, the caption of the currently selected image becomes the new global caption
            if self.preview_image_path and self.preview_image_path in self.image_settings:
                new_global_caption = self.image_settings[self.preview_image_path].get('caption', current_caption)
                
                self.updating_caption_box = True
                self.caption_text_box.delete("1.0", tk.END)
                self.caption_text_box.insert("1.0", new_global_caption)
                self.caption_text_box.edit_modified(False)
                self.updating_caption_box = False
                
                # And apply this new global caption to all images
                for path in self.image_list:
                    if path in self.image_settings:
                        self.image_settings[path]['caption'] = new_global_caption
        
        if self.preview_image_path:
            self.display_image()

    def on_tilt_toggle(self):
        if self.preview_image_path:
            # When toggling, we might need to reset angles
            if not self.random_tilt_var.get():
                for path in self.image_list:
                    if path in self.image_settings:
                        self.image_settings[path]['tilt_angle'] = 0
            self.display_image()

    def on_stroke_width_change(self, *args):
        try:
            new_width = self.stroke_width_var.get()
            self.config['stroke_width'] = new_width
            if self.preview_image_path:
                self.display_image()
        except (tk.TclError, ValueError):
            # This can happen if the spinbox is empty temporarily
            pass

    def save_config(self):
        try:
            base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)
            config_path = os.path.join(base_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            messagebox.showinfo("Settings Saved", "The configuration has been saved.")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save config.json: {e}")

    def recenter_all_captions(self):
        if not self.image_list:
            messagebox.showwarning("No Images", "There are no images to recenter.")
            return

        for path in self.image_list:
            if path in self.image_settings:
                self.image_settings[path]['manual_placement'] = False
                self.image_settings[path]['x'] = None
                self.image_settings[path]['y'] = None
        
        if self.preview_image_path:
            self.display_image()

        messagebox.showinfo("Recenter Complete", "All caption positions have been reset to automatic placement.")

    def select_output_folder(self):
        folder_selected = filedialog.askdirectory(title="Select Output Folder")
        if folder_selected:
            self.output_folder_path.set(folder_selected)

    def on_file_select(self, event=None):
        self._update_counts()
        selection = self.listbox.curselection()
        if not selection:
            self.preview_image_path = None
            self.original_image_for_preview = None
            self.preview_label.config(image=None, text="")
            self.current_preview_image = None
            return
        
        selected_index = selection[0]
        path = self.image_list[selected_index]

        if path == self.preview_image_path:
            return

        self.preview_image_path = path
        
        try:
            file_extension = os.path.splitext(path)[1].lower()
            if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']:
                self.original_image_for_preview = Image.open(self.preview_image_path).convert("RGBA")
            elif file_extension in ['.mp4', '.mov', '.avi']:
                if path in self.video_frame_cache:
                    self.original_image_for_preview = self.video_frame_cache[path]
                else:
                    self.original_image_for_preview = get_video_frame(self.preview_image_path)
                    self.video_frame_cache[path] = self.original_image_for_preview
            else:
                self.original_image_for_preview = None
                self.current_preview_image = None
                self.preview_label.config(image=None, text=f"Unsupported file type for preview\n{os.path.basename(self.preview_image_path)}")
                return
        except Exception as e:
            self.original_image_for_preview = None
            self.current_preview_image = None
            self.preview_label.config(image=None, text=f'''Cannot preview {os.path.basename(self.preview_image_path)}''')
            safe_print(f"Error loading image for preview: {e}")
            return

        caption_for_preview = self._get_caption_for_image_index(selected_index)

        if self.preview_image_path not in self.image_settings:
            self.image_settings[self.preview_image_path] = {
                'x': None, 'y': None, 'scale_x': 1.0, 'scale_y': 1.0, 
                'manual_placement': False, 'caption': '', 'wrapped_caption': ''
            }

        self.image_settings[self.preview_image_path]['caption'] = caption_for_preview

        settings = self.image_settings[self.preview_image_path]
        if isinstance(self.original_image_for_preview, Image.Image):
            img_w, _ = self.original_image_for_preview.size
        else:
            img_w, _ = self.original_image_for_preview.shape[1], self.original_image_for_preview.shape[0]

        wrapped = wrap_text(
            settings['caption'],
            resource_path(self.config['font_path']),
            self.config['font_size'],
            self.config['text_width_ratio'],
            img_w
        )
        self.image_settings[self.preview_image_path]['wrapped_caption'] = wrapped

        self.display_image()

    def _get_caption_for_image_index(self, index):
        try:
            num_groups = int(self.group_size_var.get())
        except ValueError:
            num_groups = 1

        captions_text = self.caption_text_box.get("1.0", tk.END).strip()
        captions = [c.strip() for c in re.split(r'\n\s*\n', captions_text) if c.strip()]
        
        if not captions:
            return ""

        if num_groups > 1:
            total_images = len(self.image_list)
            if total_images == 0:
                return captions[0] if captions else ""

            base_size = total_images // num_groups
            remainder = total_images % num_groups
            
            current_pos = 0
            for i in range(num_groups):
                group_size_for_this_group = base_size + (1 if i < remainder else 0)
                if current_pos <= index < current_pos + group_size_for_this_group:
                    return captions[i % len(captions)]
                current_pos += group_size_for_this_group
            
            return captions[0]
        else:
            return captions[0]

    def on_caption_change(self, event=None):
        if self.updating_caption_box:
            return

        if self.caption_update_timer:
            self.after_cancel(self.caption_update_timer)
        self.caption_update_timer = self.after(300, self._perform_caption_update)
        self.overlay_image_path = None
        self.overlay_image_label_var.set("No overlay image selected")

    def _perform_caption_update(self):
        """ Updates the preview based on the content of the caption box. """
        if not self.preview_image_path:
            self.caption_text_box.edit_modified(False)
            return

        try:
            idx = self.image_list.index(self.preview_image_path)
        except ValueError:
            self.caption_text_box.edit_modified(False)
            return

        caption_for_preview = self._get_caption_for_image_index(idx)
        
        self.image_settings[self.preview_image_path]['caption'] = caption_for_preview

        if self.original_image_for_preview and self.preview_image_path in self.image_settings:
            if isinstance(self.original_image_for_preview, Image.Image):
                img_w, _ = self.original_image_for_preview.size
            else:
                img_w, _ = self.original_image_for_preview.shape[1], self.original_image_for_preview.shape[0]

            wrapped = wrap_text(
                caption_for_preview, 
                resource_path(self.config['font_path']),
                self.config['font_size'], 
                self.config['text_width_ratio'], 
                img_w
            )
            self.image_settings[self.preview_image_path]['wrapped_caption'] = wrapped

        self.display_image()
        self.caption_text_box.edit_modified(False)

    def display_image(self):
        if self.original_image_for_preview is None:
            self.preview_label.config(image=None, text="No image selected or image failed to load.")
            return

        image_path = self.preview_image_path
        try:
            image_to_display = self.original_image_for_preview.copy()
            settings = self.image_settings.get(image_path, {})

            # Generate the captioned image using the centralized function
            image_to_display, bbox = generate_captioned_image(
                image_to_display, 
                settings, 
                self.config, 
                self.face_detector, 
                self.random_tilt_var.get(),
                self.font_outline_var.get(),
                overlay_image_path=self.overlay_image_path
            )

            panel_width = self.preview_frame.winfo_width()
            panel_height = self.preview_frame.winfo_height()
            
            if panel_width < 2 or panel_height < 2:
                self.after(50, lambda: self.display_image())
                return

            img_w, img_h = image_to_display.size
            ratio = min(panel_width / img_w, panel_height / img_h)
            new_w, new_h = int(img_w * ratio), int(img_h * ratio)
            
            resized_image = image_to_display.resize((new_w, new_h), Image.LANCZOS)
            
            if bbox:
                x, y, tw, th = bbox
                preview_x = x * ratio
                preview_y = y * ratio
                preview_tw = tw * ratio
                preview_th = th * ratio

                offset_x = (panel_width - new_w) / 2
                offset_y = (panel_height - new_h) / 2
                
                self.caption_bbox_on_preview = (preview_x + offset_x, preview_y + offset_y, preview_x + offset_x + preview_tw, preview_y + offset_y + preview_th)

                handle_size = 10
                handle_x = self.caption_bbox_on_preview[2]
                handle_y = self.caption_bbox_on_preview[3]
                self.resize_handle_bbox = (handle_x - handle_size/2, handle_y - handle_size/2, handle_x + handle_size/2, handle_y + handle_size/2)
                
                preview_draw = ImageDraw.Draw(resized_image)
                handle_draw_x1 = self.resize_handle_bbox[0] - offset_x
                handle_draw_y1 = self.resize_handle_bbox[1] - offset_y
                handle_draw_x2 = self.resize_handle_bbox[2] - offset_x
                handle_draw_y2 = self.resize_handle_bbox[3] - offset_y
                preview_draw.rectangle(
                    (handle_draw_x1, handle_draw_y1, handle_draw_x2, handle_draw_y2),
                    fill="red", outline="white"
                )
            else:
                self.caption_bbox_on_preview = None
                self.resize_handle_bbox = None

            self.current_preview_image = ImageTk.PhotoImage(resized_image)
            self.preview_label.config(image=self.current_preview_image)
            
        except Exception as e:
            self.preview_label.config(image=None, text=f'''Cannot preview {os.path.basename(image_path)}''')
            safe_print(f"Error displaying image: {e}")

    def start_bulk_preview(self):
        # Determine which items to preview: selected images, or all images if none selected
        selected_indices = self.listbox.curselection()
        all_paths = [self.image_list[i] for i in selected_indices] if selected_indices else self.image_list[:]
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        video_exts = ['.mp4', '.mov', '.avi']
        preview_paths = [p for p in all_paths if os.path.splitext(p)[1].lower() in image_exts + video_exts]

        if not preview_paths:
            messagebox.showwarning("No Media", "No images or videos selected or available for preview.")
            return

        # Create a simple scrollable popup
        top = tk.Toplevel(self)
        top.title("Bulk Preview")
        # Widen the window a bit to accommodate larger thumbnails in 10 columns
        top.geometry("900x600")

        canvas = tk.Canvas(top)
        scrollbar = tk.Scrollbar(top, orient=tk.VERTICAL, command=canvas.yview)
        container = tk.Frame(canvas)
        container.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        # Keep a handle to the inner window so we can match widths to avoid clipping
        window_id = canvas.create_window((0, 0), window=container, anchor="nw")
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(window_id, width=e.width)
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Keep references to PhotoImage objects
        self._bulk_preview_photos = []

        status_var = tk.StringVar(value=f"Rendering 0/{len(preview_paths)} previews...")
        status_label = tk.Label(top, textvariable=status_var, anchor='w')
        status_label.pack(fill=tk.X, padx=8, pady=4)

        def render_one(i=0):
            if i >= len(preview_paths):
                status_var.set(f"Rendered {len(preview_paths)} preview(s).")
                return
            path = preview_paths[i]
            try:
                ext = os.path.splitext(path)[1].lower()
                if ext in image_exts:
                    with Image.open(path) as base_img:
                        base = base_img.convert("RGBA")
                else:
                    base = get_video_frame(path)
                    if base is None:
                        raise RuntimeError("Could not extract video preview frame")

                # Apply resize option if enabled to reflect final output
                if self.resize_enabled.get():
                    try:
                        w = int(self.width_var.get())
                        h = int(self.height_var.get())
                        if w > 0 and h > 0:
                            base = resize_and_crop(base, w, h)
                    except Exception:
                        pass

                # Build settings per image
                settings = self.image_settings.get(path, {}).copy()

                # Use the same per-image caption logic as the main preview/final output
                # Supports multiple captions separated by blank lines and grouping
                try:
                    idx_in_all = self.image_list.index(path)
                except ValueError:
                    idx_in_all = 0
                settings['caption'] = self._get_caption_for_image_index(idx_in_all)

                # Ensure wrapped caption is present based on current config and image width
                wrapped = wrap_text(
                    settings.get('caption', ''),
                    resource_path(self.config['font_path']),
                    self.config['font_size'],
                    self.config['text_width_ratio'],
                    base.size[0]
                )
                settings['wrapped_caption'] = wrapped

                final_img, _ = generate_captioned_image(
                    base,
                    settings,
                    self.config,
                    self.face_detector,
                    self.random_tilt_var.get(),
                    self.font_outline_var.get(),
                    overlay_image_path=self.overlay_image_path
                )

                # Create a slightly larger thumbnail for 10-per-row grid
                # 80px width with window widened to 900 keeps 10 columns fitting
                max_w = 80
                ratio = min(1.0, max_w / float(final_img.size[0]))
                thumb_size = (int(final_img.size[0] * ratio), int(final_img.size[1] * ratio))
                thumb = final_img.resize(thumb_size, Image.LANCZOS)

                photo = ImageTk.PhotoImage(thumb)
                self._bulk_preview_photos.append(photo)

                # Grid layout: 10 items per row
                columns = 10
                row = i // columns
                col = i % columns

                item_frame = tk.Frame(container, bd=1, relief=tk.SOLID)
                item_frame.grid(row=row, column=col, padx=4, pady=4, sticky="n")

                # Add a small number label above the image
                num_label = tk.Label(item_frame, text=str(i+1))
                num_label.pack(padx=3, pady=(3, 0))

                # Show the image to keep cells compact
                img_label = tk.Label(item_frame, image=photo)
                img_label.pack(padx=3, pady=3)

            except Exception as e:
                # Place error cell in the same grid position
                columns = 10
                row = i // columns
                col = i % columns
                err_frame = tk.Frame(container, bd=1, relief=tk.SOLID)
                err_frame.grid(row=row, column=col, padx=4, pady=4, sticky="n")
                tk.Label(err_frame, text=str(i+1)).pack(padx=3, pady=(3, 0))
                tk.Label(err_frame, text=f"Error", fg='red').pack(padx=3, pady=(0, 0))
                tk.Label(err_frame, text=os.path.basename(path), wraplength=96, justify='center').pack(padx=3, pady=3)
            finally:
                status_var.set(f"Rendering {i+1}/{len(preview_paths)} previews...")
                top.after(10, render_one, i + 1)

        # Kick off progressive rendering so the window shows up immediately
        top.after(10, render_one)

    

    def add_media(self):
        files = filedialog.askopenfilenames(
            title="Select Images or Videos",
            filetypes=(("Media Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp *.mp4 *.mov *.avi"), ("All files", "*.*"))
        )
        if not files: return

        caption_text = self.caption_text_box.get("1.0", tk.END).strip()
        added_new = False
        for f in files:
            if f not in self.image_list:
                self.image_list.append(f)
                added_new = True
                if f not in self.image_settings:
                    self.image_settings[f] = {'x': None, 'y': None, 'scale_x': 1.0, 'scale_y': 1.0, 'manual_placement': False, 'caption': caption_text}
        
        if added_new:
            self._update_listbox()

        if self.listbox.size() > 0 and not self.listbox.curselection():
            self.listbox.select_set(0)
            self.on_file_select()

    def select_all_images(self):
        self.listbox.select_set(0, tk.END)
        self._update_counts()
        # Trigger on_file_select to update preview if necessary
        if self.listbox.size() > 0:
            self.on_file_select()

    def remove_selected(self):
        selected_indices = self.listbox.curselection()
        if not selected_indices:
            return

        # Get paths to remove before modifying the list
        paths_to_remove = [self.image_list[i] for i in selected_indices]

        for path in paths_to_remove:
            if path in self.image_list:
                self.image_list.remove(path)
            if path in self.image_settings:
                del self.image_settings[path]

        self._update_listbox()

        # After deletion, update the preview. If no items left, clear preview.
        if self.listbox.size() == 0:
            self.preview_label.config(image=None, text="")
            self.original_image_for_preview = None
            self.current_preview_image = None
            self.preview_image_path = None
        else:
            # Try to select the first item if nothing is selected, or the next available item
            if not self.listbox.curselection():
                if self.listbox.size() > 0:
                    self.listbox.select_set(0)
            self.on_file_select()

    def start_processing(self):
        files_to_process = [file for file in self.image_list if os.path.splitext(file)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']]

        if not files_to_process:
            messagebox.showwarning("No Images", "No images found in the list.")
            return

        output_folder = self.output_folder_path.get()
        if not output_folder:
            messagebox.showwarning("No Output Folder", "Please select an output folder.")
            return
        
        try:
            group_size = int(self.group_size_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Group size must be a valid number.")
            return

        captions_text = self.caption_text_box.get("1.0", tk.END).strip()
        captions = [c.strip() for c in re.split(r'\n\s*\n', captions_text) if c.strip()]
        separate_folders = self.separate_folders_var.get()

        self.start_button.config(state=tk.DISABLED)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        self.progress_bar['value'] = 0
        self.update_idletasks()

        thread = threading.Thread(target=self._process_images_thread, args=(files_to_process, output_folder, group_size, captions, separate_folders))
        thread.start()
        self._update_progress_from_queue("images")

    def _process_images_thread(self, files_to_process, output_folder, group_size, captions, separate_folders):
        try:
            num_groups = group_size
            total_images = len(files_to_process)

            if num_groups > 1 and total_images > 0:
                base_size = total_images // num_groups
                remainder = total_images % num_groups
                image_groups = []
                current_pos = 0
                for i in range(num_groups):
                    group_size_for_this_group = base_size + (1 if i < remainder else 0)
                    image_groups.append(files_to_process[current_pos:current_pos + group_size_for_this_group])
                    current_pos += group_size_for_this_group
            else:
                image_groups = [files_to_process]
                captions = [captions[0] if captions else '']

            resolution = None
            if self.resize_enabled.get():
                try:
                    width = int(self.width_var.get())
                    height = int(self.height_var.get())
                    if width > 0 and height > 0:
                        resolution = (width, height)
                    else:
                        raise ValueError()
                except ValueError:
                    self.progress_queue.put(("error", "Invalid Resolution: Please enter valid, positive integers for width and height."))
                    return

            files_processed_count = 0
            total_files_count = len(files_to_process)

            for i, group in enumerate(image_groups):
                if captions:
                    group_caption = captions[i % len(captions)]
                else:
                    group_caption = ""
                group_output_folder = output_folder
                
                if separate_folders and group_size > 0:
                    start_num = i * group_size + 1
                    end_num = start_num + len(group) - 1
                    folder_name = f"group_{start_num}-{end_num}"
                    group_output_folder = os.path.join(output_folder, folder_name)

                for path in group:
                    try:
                        # This is inefficient but matches original design
                        with Image.open(path) as img:
                            img_w, _ = img.size
                            wrapped = wrap_text(
                                group_caption,
                                resource_path(self.config['font_path']),
                                self.config['font_size'],
                                self.config['text_width_ratio'],
                                img_w
                            )
                        
                        # Use existing settings if available, but update caption
                        if path not in self.image_settings:
                             self.image_settings[path] = {'x': None, 'y': None, 'scale_x': 1.0, 'scale_y': 1.0, 'manual_placement': False}
                        self.image_settings[path]['caption'] = group_caption
                        self.image_settings[path]['wrapped_caption'] = wrapped
                        if self.overlay_image_path:
                            self.image_settings[path]['overlay_image_path'] = self.overlay_image_path

                    except Exception as e:
                        safe_print(f"Could not process/wrap text for {path}: {e}")

                process_images(
                    image_paths=group,
                    output_folder=group_output_folder,
                    font_path=resource_path(self.config['font_path']),
                    font_size=self.config['font_size'],
                    text_width_ratio=self.config['text_width_ratio'],
                    text_color=tuple(self.config['text_color']),
                    stroke_color=tuple(self.config['stroke_color']),
                    stroke_width=self.config['stroke_width'],
                    resolution=resolution,
                    image_settings=self.image_settings,
                    progress_callback=lambda current, total: self.progress_queue.put(((files_processed_count + current) / total_files_count) * 100),
                    random_tilt=self.random_tilt_var.get(),
                    font_outline=self.font_outline_var.get(),
                    filename_option=self.filename_option.get(),
                    start_index=files_processed_count
                )
                files_processed_count += len(group)

            self.progress_queue.put("done_images")
        except Exception as e:
            import traceback
            import io
            s = io.StringIO()
            traceback.print_exc(file=s)
            self.progress_queue.put(("error", f"An unexpected error occurred: {e}\n{s.getvalue()}"))

    def start_video_processing(self):
        video_paths = [file for file in self.image_list if os.path.splitext(file)[1].lower() in ['.mp4', '.mov', '.avi']]

        if not video_paths:
            messagebox.showwarning("No Videos", "No videos found in the list.")
            return

        output_folder = self.output_folder_path.get()
        if not output_folder:
            messagebox.showwarning("No Output Folder", "Please select an output folder.")
            return

        resolution = None
        if self.resize_enabled.get():
            try:
                width = int(self.width_var.get())
                height = int(self.height_var.get())
                if width > 0 and height > 0:
                    resolution = (width, height)
                else:
                    raise ValueError()
            except ValueError:
                messagebox.showerror("Invalid Resolution", "Please enter valid, positive integers for width and height.")
                return

        self.process_video_button.config(state=tk.DISABLED)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        self.progress_bar['value'] = 0
        self.update_idletasks()

        thread = threading.Thread(target=self._process_videos_thread, args=(video_paths, output_folder, resolution))
        thread.start()
        self._update_progress_from_queue("videos")

    def _process_videos_thread(self, video_paths, output_folder, resolution):
        total_videos = len(video_paths)
        for i, path in enumerate(video_paths):
            try:
                from image_processor import process_video
                if path not in self.image_settings:
                    self.image_settings[path] = {}
                if self.overlay_image_path:
                    self.image_settings[path]['overlay_image_path'] = self.overlay_image_path
                process_video(
                    video_path=path,
                    output_folder=output_folder,
                    font_path=resource_path(self.config['font_path']),
                    font_size=self.config['font_size'],
                    text_width_ratio=self.config['text_width_ratio'],
                    text_color=tuple(self.config['text_color']),
                    stroke_color=tuple(self.config['stroke_color']),
                    stroke_width=self.config['stroke_width'],
                    resolution=resolution,
                    image_settings=self.image_settings,
                    progress_callback=lambda current, total: self.progress_queue.put(((i + (current / total)) / total_videos) * 100),
                    random_tilt=self.random_tilt_var.get(),
                    font_outline=self.font_outline_var.get(),
                    filename_option=self.filename_option.get(),
                    face_detector=self.face_detector
                )
            except Exception as e:
                self.progress_queue.put(("error", f"An error occurred while processing {os.path.basename(path)}: {e}"))

        self.progress_queue.put("done_videos")

    def _update_progress_from_queue(self, process_type):
        try:
            progress = self.progress_queue.get_nowait()

            if isinstance(progress, tuple) and progress[0] == "error":
                _, error_message = progress
                if "\n" in error_message: # Check if it's a detailed traceback
                    self.show_error_popup(error_message)
                else:
                    messagebox.showerror("Error", error_message)
                
                if process_type == "images":
                    self.start_button.config(state=tk.NORMAL)
                elif process_type == "videos":
                    self.process_video_button.config(state=tk.NORMAL)
                self.progress_bar.pack_forget()
                return # Stop polling

            if progress == "done_images":
                self.start_button.config(state=tk.NORMAL)
                self.progress_bar.pack_forget()
                messagebox.showinfo("Success", "Image processing complete!")
            elif progress == "done_videos":
                self.process_video_button.config(state=tk.NORMAL)
                self.progress_bar.pack_forget()
                messagebox.showinfo("Success", "Video processing complete!")
            else:
                self.progress_bar['value'] = progress
                self.after(100, lambda: self._update_progress_from_queue(process_type))
        except queue.Empty:
            self.after(100, lambda: self._update_progress_from_queue(process_type))

if __name__ == "__main__":
    app = App()
    app.mainloop()
