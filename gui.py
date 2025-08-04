import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
import shutil
from PIL import Image, ImageTk, ImageDraw
from image_processor import process_images, resource_path

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Captioner")
        self.geometry("1000x700")

        # --- Data ---
        self.manual_placements = {}
        self.current_preview_image = None
        self.original_image_for_preview = None
        self.preview_image_path = None

        # --- Load Config ---
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
        self.preview_label.bind('<Button-1>', self.on_image_click)

        # Right Panel (Controls)
        self.control_frame = tk.Frame(self.main_frame, width=200)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self.control_frame.pack_propagate(False)

        # Placement Mode
        self.mode_frame = tk.LabelFrame(self.control_frame, text="Placement Mode")
        self.mode_frame.pack(pady=10, fill=tk.X)
        self.placement_mode = tk.StringVar(value="Automatic")
        tk.Radiobutton(self.mode_frame, text="Automatic", variable=self.placement_mode, value="Automatic").pack(anchor=tk.W)
        tk.Radiobutton(self.mode_frame, text="Manual - Same for All", variable=self.placement_mode, value="Manual - Same for All").pack(anchor=tk.W)
        tk.Radiobutton(self.mode_frame, text="Manual - Individual", variable=self.placement_mode, value="Manual - Individual").pack(anchor=tk.W)

        self.start_button = tk.Button(self.control_frame, text="Start Processing", command=self.start_processing)
        self.start_button.pack(pady=20, fill=tk.X)

    def on_file_select(self, event=None):
        selection = self.listbox.curselection()
        if not selection:
            return
        
        selected_index = selection[0]
        self.preview_image_path = self.listbox.get(selected_index)
        
        self.display_image(self.preview_image_path)

    def display_image(self, image_path, click_pos=None):
        try:
            self.original_image_for_preview = Image.open(image_path).convert("RGBA")
            
            # Draw on a copy
            image_to_display = self.original_image_for_preview.copy()

            # If a click position is provided, draw a marker
            if click_pos:
                draw = ImageDraw.Draw(image_to_display)
                r = 5
                draw.ellipse((click_pos[0]-r, click_pos[1]-r, click_pos[0]+r, click_pos[1]+r), fill='red', outline='white')

            # Resize for preview
            panel_width = self.preview_frame.winfo_width()
            panel_height = self.preview_frame.winfo_height()
            
            if panel_width < 2 or panel_height < 2: # Frame not rendered yet
                self.after(50, lambda: self.display_image(image_path, click_pos))
                return

            img_w, img_h = image_to_display.size
            ratio = min(panel_width / img_w, panel_height / img_h)
            new_w, new_h = int(img_w * ratio), int(img_h * ratio)
            
            resized_image = image_to_display.resize((new_w, new_h), Image.LANCZOS)
            
            self.current_preview_image = ImageTk.PhotoImage(resized_image)
            self.preview_label.config(image=self.current_preview_image)
            
        except Exception as e:
            self.preview_label.config(image=None, text=f"""Cannot preview
{os.path.basename(image_path)}""")
            print(f"Error displaying image: {e}")

    def on_image_click(self, event):
        if not self.original_image_for_preview or not self.preview_image_path:
            return

        # Get click relative to the preview label
        click_x, click_y = event.x, event.y

        # Get the size of the displayed (resized) image
        resized_w = self.current_preview_image.width()
        resized_h = self.current_preview_image.height()
        
        # Get the size of the label
        label_w = self.preview_label.winfo_width()
        label_h = self.preview_label.winfo_height()

        # Calculate the padding (if image is centered)
        pad_x = (label_w - resized_w) // 2
        pad_y = (label_h - resized_h) // 2

        # Check if click is outside the actual image area
        if not (pad_x <= click_x < pad_x + resized_w and pad_y <= click_y < pad_y + resized_h):
            return

        # Adjust click coordinates to be relative to the top-left of the image
        adjusted_x = click_x - pad_x
        adjusted_y = click_y - pad_y

        # Convert preview coordinates to original image coordinates
        original_w, original_h = self.original_image_for_preview.size
        ratio = original_w / resized_w
        
        original_x = int(adjusted_x * ratio)
        original_y = int(adjusted_y * ratio)

        # Store the coordinate
        self.manual_placements[self.preview_image_path] = (original_x, original_y)
        
        # Redraw the preview with a marker
        self.display_image(self.preview_image_path, click_pos=(original_x, original_y))
        
        messagebox.showinfo("Position Set", f"Caption position for {os.path.basename(self.preview_image_path)} set to ({original_x}, {original_y}).")


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

        selected_path = self.listbox.get(selected_indices[0])
        
        # Remove from placements dict
        if selected_path in self.manual_placements:
            del self.manual_placements[selected_path]
            
        self.listbox.delete(selected_indices[0])

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

        output_folder = self.config['output_folder']
        if os.path.exists(output_folder):
            try:
                shutil.rmtree(output_folder)
                print(f"Cleared existing output folder: {output_folder}")
            except OSError as e:
                messagebox.showerror("Error", f"Error clearing output folder: {e}")
                return

        mode = self.placement_mode.get()
        
        # Validate manual placements if needed
        if mode == "Manual - Same for All" and not self.manual_placements:
            messagebox.showwarning("No Position Set", "Please select an image and click on it to set the caption position first.")
            return
        if mode == "Manual - Individual":
            missing_placements = [os.path.basename(f) for f in files_to_process if f not in self.manual_placements]
            if missing_placements:
                messagebox.showwarning("Missing Positions", f"""The following images are missing a manual caption position:

{', '.join(missing_placements)}""")
                return

        try:
            with open(resource_path('caption.txt'), 'r', encoding='utf-8') as f:
                caption_text = f.read().strip()

            process_images(
                image_paths=files_to_process,
                output_folder=self.config['output_folder'],
                caption_text=caption_text,
                font_path=resource_path(self.config['font_path']),
                font_size_divisor=self.config['font_size_divisor'],
                text_width_ratio=self.config['text_width_ratio'],
                text_color=tuple(self.config['text_color']),
                stroke_color=tuple(self.config['stroke_color']),
                stroke_width=self.config['stroke_width'],
                # --- New parameters ---
                placement_mode=mode,
                manual_placements=self.manual_placements
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