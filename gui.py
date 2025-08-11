import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
import os
import shutil
from PIL import Image, ImageTk, ImageDraw
import mediapipe as mp
from image_processor import process_images, resource_path, multiline_bbox, draw_caption, get_automatic_placement_coords, safe_print

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Captioner")
        self.geometry("1000x700")

        # --- Data ---
        self.current_preview_image = None
        self.original_image_for_preview = None
        self.preview_image_path = None

        self.image_settings = {}  # path -> {'x': None, 'y': None, 'scale_x': 1.0, 'scale_y': 1.0, 'manual_placement': False, 'caption': ''}
        self.drag_info = {'start_x_offset': 0, 'start_y_offset': 0, 'is_dragging': False}
        self.resize_info = {'is_resizing': False, 'start_bbox': (0,0,0,0), 'start_scale_x': 1.0, 'start_scale_y': 1.0}
        self.caption_bbox_on_preview = None
        self.resize_handle_bbox = None
        self.updating_caption_box = False

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        with open(resource_path('config.json'), 'r') as f:
            self.config = json.load(f)

        # --- UI Elements ---
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Left Panel (File List)
        self.list_frame = tk.Frame(self.main_frame, width=200)
        self.list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.list_frame.pack_propagate(False)
        self.add_button = tk.Button(self.list_frame, text="Add Images", command=self.add_images)
        self.add_button.pack(pady=5, fill=tk.X)
        self.remove_button = tk.Button(self.list_frame, text="Remove Selected", command=self.remove_selected)
        self.remove_button.pack(pady=5, fill=tk.X)
        self.listbox = tk.Listbox(self.list_frame, selectmode=tk.SINGLE)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self.listbox.bind('<<ListboxSelect>>', self.on_file_select)

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
        self.sync_frame.pack(pady=10, fill=tk.X)
        self.apply_to_all_var = tk.BooleanVar(value=True)
        self.apply_to_all_checkbox = tk.Checkbutton(self.sync_frame, text="Apply position/scale to all", variable=self.apply_to_all_var)
        self.apply_to_all_checkbox.pack(anchor=tk.W, padx=5, pady=2)
        self.sync_caption_var = tk.BooleanVar(value=True)
        self.sync_caption_checkbox = tk.Checkbutton(self.sync_frame, text="Use one caption for all", variable=self.sync_caption_var)
        self.sync_caption_checkbox.pack(anchor=tk.W, padx=5, pady=2)
        self.recenter_button = tk.Button(self.sync_frame, text="Recenter All Captions", command=self.recenter_all_captions)
        self.recenter_button.pack(fill=tk.X, padx=5, pady=(2, 5))
        self.sync_caption_var.trace_add('write', self.on_sync_caption_toggle)

        # Caption Input
        self.caption_frame = tk.LabelFrame(self.control_frame, text="Caption Text")
        self.caption_frame.pack(pady=10, fill=tk.X)
        self.caption_text_box = tk.Text(self.caption_frame, height=4)
        self.caption_text_box.pack(pady=5, padx=5, fill=tk.X)
        self.caption_text_box.bind('<<Modified>>', self.on_caption_change)

        # Output Folder
        self.output_folder_frame = tk.LabelFrame(self.control_frame, text="Output Folder")
        self.output_folder_frame.pack(pady=10, fill=tk.X)
        self.output_folder_path = tk.StringVar(value=self.config.get('output_folder', 'captioned_images'))
        self.output_folder_entry = tk.Entry(self.output_folder_frame, textvariable=self.output_folder_path)
        self.output_folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.browse_button = tk.Button(self.output_folder_frame, text="...", command=self.select_output_folder)
        self.browse_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # Resolution Settings
        self.resolution_frame = tk.LabelFrame(self.control_frame, text="Resize Options")
        self.resolution_frame.pack(pady=10, fill=tk.X)
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

        self.start_button = tk.Button(self.control_frame, text="Start Processing", command=self.start_processing)
        self.start_button.pack(pady=20, fill=tk.X)

        self.progress_bar = ttk.Progressbar(self.control_frame, orient='horizontal', mode='determinate')

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
        if not self.preview_image_path: return

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
                for path in self.listbox.get(0, tk.END):
                    if path not in self.image_settings:
                        self.image_settings[path] = {'x': None, 'y': None, 'scale_x': 1.0, 'scale_y': 1.0, 'manual_placement': False, 'caption': ''}
                    self.image_settings[path]['scale_x'] = new_scale_x
                    self.image_settings[path]['scale_y'] = new_scale_y
            else:
                settings['scale_x'] = new_scale_x
                settings['scale_y'] = new_scale_y
            
            self.display_image(self.preview_image_path)

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
                for path in self.listbox.get(0, tk.END):
                    if path not in self.image_settings:
                        self.image_settings[path] = {'x': None, 'y': None, 'scale_x': 1.0, 'scale_y': 1.0, 'manual_placement': False, 'caption': ''}
                    self.image_settings[path]['x'] = new_x
                    self.image_settings[path]['y'] = new_y
                    self.image_settings[path]['manual_placement'] = True
            else:
                settings['x'] = new_x
                settings['y'] = new_y
                settings['manual_placement'] = True
            
            self.display_image(self.preview_image_path)

    def on_mouse_release(self, event):
        self.drag_info['is_dragging'] = False
        self.resize_info['is_resizing'] = False

    def update_progress(self, current, total):
        self.progress_bar['value'] = (current / total) * 100
        self.update_idletasks()

    def on_sync_caption_toggle(self, *args):
        is_synced = self.sync_caption_var.get()
        current_caption = self.caption_text_box.get("1.0", tk.END).strip()

        if not is_synced:
            # When switching from SYNCED to UNSYNCED, copy the current "global" caption to all individual images
            for path in self.listbox.get(0, tk.END):
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
                for path in self.listbox.get(0, tk.END):
                    if path in self.image_settings:
                        self.image_settings[path]['caption'] = new_global_caption
        
        if self.preview_image_path:
            self.display_image(self.preview_image_path)

    def recenter_all_captions(self):
        paths = self.listbox.get(0, tk.END)
        if not paths:
            messagebox.showwarning("No Images", "There are no images to recenter.")
            return

        for path in paths:
            if path in self.image_settings:
                self.image_settings[path]['manual_placement'] = False
                self.image_settings[path]['x'] = None
                self.image_settings[path]['y'] = None
        
        if self.preview_image_path:
            self.display_image(self.preview_image_path)

        messagebox.showinfo("Recenter Complete", "All caption positions have been reset to automatic placement.")

    def select_output_folder(self):
        folder_selected = filedialog.askdirectory(title="Select Output Folder")
        if folder_selected:
            self.output_folder_path.set(folder_selected)

    def on_file_select(self, event=None):
        selection = self.listbox.curselection()
        if not selection:
            return
        
        selected_index = selection[0]
        self.preview_image_path = self.listbox.get(selected_index)
        
        # Get the global caption before initializing settings for a new image
        global_caption = self.caption_text_box.get("1.0", tk.END).strip()

        if self.preview_image_path not in self.image_settings:
            self.image_settings[self.preview_image_path] = {
                'x': None, 'y': None, 'scale_x': 1.0, 'scale_y': 1.0, 
                'manual_placement': False, 'caption': global_caption
            }

        # If not syncing, update the textbox to show the image's specific caption
        if not self.sync_caption_var.get():
            self.updating_caption_box = True # Prevent on_caption_change from firing
            self.caption_text_box.delete("1.0", tk.END)
            self.caption_text_box.insert("1.0", self.image_settings[self.preview_image_path].get('caption', ''))
            self.caption_text_box.edit_modified(False) # Reset modified flag
            self.updating_caption_box = False

        self.display_image(self.preview_image_path)

    def on_caption_change(self, event=None):
        if self.updating_caption_box:
            return

        caption_text = self.caption_text_box.get("1.0", tk.END).strip()

        if self.sync_caption_var.get():
            # Sync to all images
            for path in self.listbox.get(0, tk.END):
                if path in self.image_settings:
                    self.image_settings[path]['caption'] = caption_text
        
        # Always update the current image's setting
        if self.preview_image_path and self.preview_image_path in self.image_settings:
                self.image_settings[self.preview_image_path]['caption'] = caption_text

        if self.preview_image_path:
            self.display_image(self.preview_image_path)
        
        self.caption_text_box.edit_modified(False)

    def display_image(self, image_path):
        try:
            self.original_image_for_preview = Image.open(image_path).convert("RGBA")
            image_to_display = self.original_image_for_preview.copy()

            settings = self.image_settings.get(image_path, {})
            caption_text = settings.get('caption', '')
            scale_x = settings.get('scale_x', 1.0)
            scale_y = settings.get('scale_y', 1.0)

            wrapped_caption = ""
            font = None
            tw, th = 0, 0

            if caption_text:
                draw = ImageDraw.Draw(image_to_display)
                font_path = resource_path(self.config['font_path'])
                font_size_divisor = self.config['font_size_divisor']
                text_width_ratio = self.config['text_width_ratio']
                text_color = tuple(self.config['text_color'])
                stroke_color = tuple(self.config['stroke_color'])
                base_stroke_width = self.config['stroke_width']

                # Use an estimated average scale for font size and stroke to maintain visual consistency
                avg_scale = (scale_x + scale_y) / 2
                scaled_font_size_divisor = font_size_divisor / avg_scale
                scaled_stroke_width = int(base_stroke_width * avg_scale)

                # Calculate the target width and height for the caption based on scaling
                target_caption_width = image_to_display.width * text_width_ratio * scale_x

                lines, block_height, font = draw_caption(
                    draw, caption_text, font_path, scaled_font_size_divisor,
                    text_width_ratio, text_color, stroke_color, scaled_stroke_width,
                    image_to_display.width, image_to_display.height,
                    target_caption_width=target_caption_width
                )
                wrapped_caption = "\n".join(lines)
                
                tw, th = multiline_bbox(draw, wrapped_caption, font, stroke_w=scaled_stroke_width)

                # Placement Logic
                is_manual = settings.get('manual_placement', False)

                if not is_manual:
                    # Always recenter horizontally in auto mode
                    settings['x'] = (image_to_display.width - tw) / 2
                
                # Set initial Y position if not set yet (for both modes)
                if settings.get('y') is None:
                    settings['y'] = get_automatic_placement_coords(image_path, image_to_display.width, image_to_display.height, block_height, self.face_detector)

                self.image_settings[image_path] = settings
                x, y = settings['x'], settings['y']
                
                draw.multiline_text(
                    (x, y), wrapped_caption, font=font, fill=text_color,
                    stroke_width=scaled_stroke_width, stroke_fill=stroke_color,
                    spacing=4, align="center"
                )

            panel_width = self.preview_frame.winfo_width()
            panel_height = self.preview_frame.winfo_height()
            
            if panel_width < 2 or panel_height < 2:
                self.after(50, lambda: self.display_image(image_path))
                return

            img_w, img_h = image_to_display.size
            ratio = min(panel_width / img_w, panel_height / img_h)
            new_w, new_h = int(img_w * ratio), int(img_h * ratio)
            
            resized_image = image_to_display.resize((new_w, new_h), Image.LANCZOS)
            
            if caption_text:
                preview_x = settings.get('x', 0) * ratio
                preview_y = settings.get('y', 0) * ratio
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
            self.preview_label.config(image=None, text=f"""Cannot preview {os.path.basename(image_path)}""")
            safe_print(f"Error displaying image: {e}")

    def add_images(self):
        files = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=(("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp"), ("All files", "*.*" ))
        )
        if not files: return

        caption_text = self.caption_text_box.get("1.0", tk.END).strip()
        for f in files:
            if f not in self.listbox.get(0, tk.END):
                self.listbox.insert(tk.END, f)
                if f not in self.image_settings:
                    self.image_settings[f] = {'x': None, 'y': None, 'scale_x': 1.0, 'scale_y': 1.0, 'manual_placement': False, 'caption': caption_text}
        
        if self.listbox.size() > 0 and not self.listbox.curselection():
            self.listbox.select_set(0)
            self.on_file_select()

    def remove_selected(self):
        selected_indices = self.listbox.curselection()
        if not selected_indices:
            return

        selected_index = selected_indices[0]
        path_to_remove = self.listbox.get(selected_index)

        self.listbox.delete(selected_index)

        if path_to_remove in self.image_settings:
            del self.image_settings[path_to_remove]

        if self.listbox.size() == 0:
            self.preview_label.config(image=None, text="")
            self.original_image_for_preview = None
            self.current_preview_image = None
            self.preview_image_path = None
        else:
            new_selection_index = min(selected_index, self.listbox.size() - 1)
            if new_selection_index >= 0:
                self.listbox.select_set(new_selection_index)
            self.on_file_select()

    def start_processing(self):
        files_to_process = self.listbox.get(0, tk.END)
        if not files_to_process:
            messagebox.showwarning("No Files", "Please add images to the list first.")
            return

        output_folder = self.output_folder_path.get()
        if not output_folder:
            messagebox.showwarning("No Output Folder", "Please select an output folder.")
            return

        self.start_button.config(state=tk.DISABLED)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        self.progress_bar['value'] = 0
        self.update_idletasks()

        if os.path.exists(output_folder):
            try:
                shutil.rmtree(output_folder)
                safe_print(f"Cleared existing output folder: {output_folder}")
            except OSError as e:
                messagebox.showerror("Error", f"Error clearing output folder: {e}")
                self.start_button.config(state=tk.NORMAL)
                self.progress_bar.pack_forget()
                return

        try:
            # Ensure all images have up-to-date caption info before processing
            current_caption = self.caption_text_box.get("1.0", tk.END).strip()
            if self.sync_caption_var.get():
                for path in files_to_process:
                    if path in self.image_settings:
                        self.image_settings[path]['caption'] = current_caption
                    else:
                         self.image_settings[path] = {'x': None, 'y': None, 'scale_x': 1.0, 'scale_y': 1.0, 'manual_placement': False, 'caption': current_caption}

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
                    self.start_button.config(state=tk.NORMAL)
                    self.progress_bar.pack_forget()
                    return

            process_images(
                image_paths=files_to_process,
                output_folder=output_folder,
                font_path=resource_path(self.config['font_path']),
                font_size_divisor=self.config['font_size_divisor'],
                text_width_ratio=self.config['text_width_ratio'],
                text_color=tuple(self.config['text_color']),
                stroke_color=tuple(self.config['stroke_color']),
                stroke_width=self.config['stroke_width'],
                resolution=resolution,
                image_settings=self.image_settings,
                progress_callback=self.update_progress
            )
            messagebox.showinfo("Success", "Image processing complete!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.start_button.config(state=tk.NORMAL)
            self.progress_bar.pack_forget()

if __name__ == "__main__":
    app = App()
    app.mainloop()