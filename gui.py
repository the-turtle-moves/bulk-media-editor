import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
from image_processor import process_images

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Captioner")
        self.geometry("800x600")

        # --- Load Config ---
        with open('config.json', 'r') as f:
            self.config = json.load(f)

        # --- UI Elements ---
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Left Panel (File Browser)
        self.file_frame = tk.Frame(self.main_frame)
        self.file_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.add_button = tk.Button(self.file_frame, text="Add Images", command=self.add_images)
        self.add_button.pack(pady=5, fill=tk.X)
        self.remove_button = tk.Button(self.file_frame, text="Remove Selected", command=self.remove_selected)
        self.remove_button.pack(pady=5, fill=tk.X)

        # Center Panel (Batch List)
        self.list_frame = tk.Frame(self.main_frame)
        self.list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox = tk.Listbox(self.list_frame, selectmode=tk.EXTENDED)
        self.listbox.pack(fill=tk.BOTH, expand=True)

        # Right Panel (Actions)
        self.action_frame = tk.Frame(self.main_frame)
        self.action_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self.start_button = tk.Button(self.action_frame, text="Start Processing", command=self.start_processing)
        self.start_button.pack(pady=5, fill=tk.X)

    def add_images(self):
        files = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=(("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*"))
        )
        for f in files:
            if f not in self.listbox.get(0, tk.END):
                self.listbox.insert(tk.END, f)

    def remove_selected(self):
        selected_indices = self.listbox.curselection()
        for i in reversed(selected_indices):
            self.listbox.delete(i)

    def start_processing(self):
        files_to_process = self.listbox.get(0, tk.END)
        if not files_to_process:
            messagebox.showwarning("No Files", "Please add images to the list first.")
            return

        try:
            with open('caption.txt', 'r', encoding='utf-8') as f:
                caption_text = f.read().strip()

            process_images(
                image_paths=files_to_process,
                output_folder=self.config['output_folder'],
                caption_text=caption_text,
                font_path=self.config['font_path'],
                font_size_divisor=self.config['font_size_divisor'],
                text_width_ratio=self.config['text_width_ratio'],
                text_color=tuple(self.config['text_color']),
                stroke_color=tuple(self.config['stroke_color']),
                stroke_width=self.config['stroke_width']
            )
            messagebox.showinfo("Success", "Image processing complete!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

if __name__ == "__main__":
    app = App()
    app.mainloop()