import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
import shutil
from PIL import Image, ImageTk, ImageDraw, ImageFont
import mediapipe as mp
import cv2
import numpy as np
from image_processor import process_images, resource_path, multiline_bbox, draw_caption, get_automatic_placement_coords
import textwrap

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Captioner")
        self.geometry("1000x700")

        # --- Data ---
        self.current_preview_image = None
        self.original_image_for_preview = None
        self.preview_image_path = None

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

        # Right Panel (Controls)
        self.control_frame = tk.Frame(self.main_frame, width=200)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self.control_frame.pack_propagate(False)

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
        self.width_var = tk.StringVar(value="1920")
        self.width_entry = tk.Entry(self.resolution_entry_frame, textvariable=self.width_var, width=7)
        self.width_entry.pack(side=tk.LEFT)
        self.height_label = tk.Label(self.resolution_entry_frame, text="Height:")
        self.height_label.pack(side=tk.LEFT, padx=5)
        self.height_var = tk.StringVar(value="1080")
        self.height_entry = tk.Entry(self.resolution_entry_frame, textvariable=self.height_var, width=7)
        self.height_entry.pack(side=tk.LEFT)

        self.start_button = tk.Button(self.control_frame, text="Start Processing", command=self.start_processing)
        self.start_button.pack(pady=20, fill=tk.X)

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
        
        self.display_image(self.preview_image_path)

    def on_caption_change(self, event=None):
        # This flag is set by Tkinter when the text is modified.
        # We need to clear it after reading the content.
        self.caption_text_box.edit_modified(False)
        if self.preview_image_path:
            self.display_image(self.preview_image_path)

    def display_image(self, image_path):
        try:
            self.original_image_for_preview = Image.open(image_path).convert("RGBA")
            
            # Draw on a copy
            image_to_display = self.original_image_for_preview.copy()

            # Get current caption text for preview
            caption_text = self.caption_text_box.get("1.0", tk.END).strip()
            if caption_text:
                draw = ImageDraw.Draw(image_to_display)
                font_path = resource_path(self.config['font_path'])
                font_size_divisor = self.config['font_size_divisor']
                text_width_ratio = self.config['text_width_ratio']
                text_color = tuple(self.config['text_color'])
                stroke_color = tuple(self.config['stroke_color'])
                stroke_width = self.config['stroke_width']

                lines, block_height, font = draw_caption(
                    draw,
                    caption_text,
                    font_path,
                    font_size_divisor,
                    text_width_ratio,
                    text_color,
                    stroke_color,
                    stroke_width,
                    image_to_display.width,
                    image_to_display.height
                )

                current_y = get_automatic_placement_coords(image_path, image_to_display.width, image_to_display.height, block_height, self.face_detector)
                wrapped_caption = "\n".join(lines)
                
                # Use a dummy draw object to calculate text size for x position
                temp_draw = ImageDraw.Draw(Image.new("RGBA", (1,1)))
                font_size = int(image_to_display.width / font_size_divisor)
                font = ImageFont.truetype(font_path, font_size)
                tw, th = multiline_bbox(temp_draw, wrapped_caption, font, stroke_w=stroke_width)
                x = (image_to_display.width - tw) / 2
                y = current_y

                draw.multiline_text(
                    (x, y),
                    wrapped_caption,
                    font=font,
                    fill=text_color,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_color,
                    spacing=4,
                    align="center"
                )

            # Resize for preview
            panel_width = self.preview_frame.winfo_width()
            panel_height = self.preview_frame.winfo_height()
            
            if panel_width < 2 or panel_height < 2: # Frame not rendered yet
                self.after(50, lambda: self.display_image(image_path))
                return

            img_w, img_h = image_to_display.size
            ratio = min(panel_width / img_w, panel_height / img_h)
            new_w, new_h = int(img_w * ratio), int(img_h * ratio)
            
            resized_image = image_to_display.resize((new_w, new_h), Image.LANCZOS)
            
            self.current_preview_image = ImageTk.PhotoImage(resized_image)
            self.preview_label.config(image=self.current_preview_image)
            
        except Exception as e:
            self.preview_label.config(image=None, text=f"""Cannot preview\n{os.path.basename(image_path)}""")
            print(f"Error displaying image: {e}")

    


    def add_images(self):
        files = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=(("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*"))
        )
        if not files: return

        for f in files:
            if f not in self.listbox.get(0, tk.END):
                self.listbox.insert(tk.END, f)
        
        # If this is the first image, select it
        if self.listbox.size() > 0 and not self.listbox.curselection():
            self.listbox.select_set(0)
            self.on_file_select()

    def remove_selected(self):
        selected_indices = self.listbox.curselection()
        if not selected_indices: return

        # Clear preview if no images are left, otherwise select the next one
        if self.listbox.size() == 0:
            self.preview_label.config(image=None, text="")
            self.original_image_for_preview = None
            self.current_preview_image = None
            self.preview_image_path = None
        else:
            new_selection_index = min(selected_indices[0], self.listbox.size() - 1)
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

        if os.path.exists(output_folder):
            try:
                shutil.rmtree(output_folder)
                print(f"Cleared existing output folder: {output_folder}")
            except OSError as e:
                messagebox.showerror("Error", f"Error clearing output folder: {e}")
                return

        

        try:
            caption_text = self.caption_text_box.get("1.0", tk.END).strip()
            if not caption_text:
                messagebox.showwarning("No Caption", "Please enter caption text.")
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

            process_images(
                image_paths=files_to_process,
                output_folder=output_folder,
                caption_text=caption_text,
                font_path=resource_path(self.config['font_path']),
                font_size_divisor=self.config['font_size_divisor'],
                text_width_ratio=self.config['text_width_ratio'],
                text_color=tuple(self.config['text_color']),
                stroke_color=tuple(self.config['stroke_color']),
                stroke_width=self.config['stroke_width'],
                resolution=resolution
            )
            messagebox.showinfo("Success", "Image processing complete!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            # Also print to console for more details
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    app = App()
    app.mainloop()