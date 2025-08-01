import os
import sys
import shutil
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import textwrap
import cv2
import mediapipe as mp
import numpy as np
import json

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def safe_print(text):
    """ A wrapper for print that checks if stdout is available. """
    if sys.stdout:
        print(text)

def process_images(image_paths, output_folder, caption_text, font_path, font_size_divisor, text_width_ratio, text_color, stroke_color, stroke_width):
    """
    Processes a list of images to add a caption.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    safe_print(f"Starting to process {len(image_paths)} images...")

    for image_path in tqdm(image_paths, desc="Captioning Images", disable=not sys.stdout):
        filename = os.path.basename(image_path)
        cv_image = cv2.imread(image_path)
        base_image = Image.open(image_path).convert("RGBA")
        draw = ImageDraw.Draw(base_image)
        
        original_height, original_width, _ = cv_image.shape

        # --- Helper functions ---
        def rotate_image(image, angle):
            height, width = image.shape[:2]
            center = (width / 2, height / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            cos = np.abs(rotation_matrix[0, 0])
            sin = np.abs(rotation_matrix[0, 1])
            new_width = int((height * sin) + (width * cos))
            new_height = int((height * cos) + (width * sin))
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]
            rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
            return rotated_image, rotation_matrix

        def get_face_detections_with_rotation(cv_image):
            angles_to_try = [0, 15, -15, 30, -30, 45, -45]
            for angle in angles_to_try:
                if angle == 0:
                    image_to_process = cv_image
                    M = None
                else:
                    image_to_process, M = rotate_image(cv_image, angle)
                rgb_image = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB)
                results = face_detector.process(rgb_image)
                if results.detections:
                    return results, M, image_to_process.shape[1], image_to_process.shape[0]
            return None, None, None, None

        # --- ADVANCED PLACEMENT LOGIC ---
        results, M, processed_width, processed_height = get_face_detections_with_rotation(cv_image)
        
        occupied_zones = []
        if results:
            if M is not None:
                inv_M = cv2.invertAffineTransform(M)
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x_min = int(bboxC.xmin * processed_width)
                y_min = int(bboxC.ymin * processed_height)
                x_max = x_min + int(bboxC.width * processed_width)
                y_max = y_min + int(bboxC.height * processed_height)
                box_corners = np.array([
                    [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]
                ], dtype=np.float32).reshape(-1, 1, 2)
                if M is not None:
                    original_corners = cv2.transform(box_corners, inv_M)
                else:
                    original_corners = box_corners
                y_coords = original_corners[:, :, 1]
                y_start = int(np.min(y_coords))
                y_end = int(np.max(y_coords))
                occupied_zones.append((y_start, y_end))

        if occupied_zones:
            occupied_zones.sort()
            merged_zones = [occupied_zones[0]]
            for current_start, current_end in occupied_zones[1:]:
                last_start, last_end = merged_zones[-1]
                if current_start <= last_end:
                    merged_zones[-1] = (last_start, max(last_end, current_end))
                else:
                    merged_zones.append((current_start, current_end))
            occupied_zones = merged_zones

        safe_zones = []
        last_y_end = 0
        for y_start, y_end in occupied_zones:
            safe_zones.append((last_y_end, y_start))
            last_y_end = y_end
        safe_zones.append((last_y_end, original_height))

        # --- CAPTIONING LOGIC ---
        font_size = int(original_width / font_size_divisor)
        font = ImageFont.truetype(font_path, font_size)
        single_line_bbox = draw.textbbox((0, 0), caption_text, font=font)
        single_line_width = single_line_bbox[2] - single_line_bbox[0]
        line_height = single_line_bbox[3] - single_line_bbox[1]
        max_text_width = original_width * text_width_ratio
        lines = [caption_text]
        if single_line_width > max_text_width:
            avg_char_width = single_line_width / len(caption_text)
            wrap_at_chars = int(max_text_width / avg_char_width) if avg_char_width > 0 else 20
            lines = textwrap.wrap(caption_text, width=wrap_at_chars)
        block_height = len(lines) * line_height
        margin = int(original_height * 0.05)
        best_zone = (0, 0)
        max_safe_height = -1
        for y_start, y_end in safe_zones:
            zone_height = y_end - y_start
            if zone_height >= block_height and zone_height > max_safe_height:
                max_safe_height = zone_height
                best_zone = (y_start, y_end)
        if max_safe_height > -1:
            zone_start, zone_end = best_zone
            zone_height = zone_end - zone_start
            offset = (zone_height - block_height) / 2
            current_y = zone_start + offset
        else:
            current_y = original_height - block_height - margin
        for line in lines:
            line_bbox = draw.textbbox((0, 0), line, font=font)
            line_width = line_bbox[2] - line_bbox[0]
            x = (original_width - line_width) / 2
            draw.text((x, current_y), line, font=font, fill=text_color, stroke_width=stroke_width, stroke_fill=stroke_color)
            current_y += line_height

        output_path = os.path.join(output_folder, filename)
        if filename.lower().endswith(('.jpg', '.jpeg')):
            base_image = base_image.convert("RGB")
        base_image.save(output_path)

    safe_print(f"\n✅ Success! All images have been captioned and saved in the '{output_folder}' folder.")



if __name__ == "__main__":
    # --- 1. LOAD CONFIGURATION ---
    with open(resource_path('config.json'), 'r') as f:
        config = json.load(f)

    input_folder = config['input_folder']
    output_folder = config['output_folder']
    font_path = resource_path(config['font_path']) # Correctly resolve font path
    font_size_divisor = config['font_size_divisor']
    text_width_ratio = config['text_width_ratio']
    text_color = tuple(config['text_color'])
    stroke_color = tuple(config['stroke_color'])
    stroke_width = config['stroke_width']

    with open(resource_path('caption.txt'), 'r', encoding='utf-8') as f:
        caption_text = f.read().strip()

    # Note: This part will only work when running as a script, not from the EXE
    if os.path.exists(input_folder):
        image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if image_files:
            process_images(image_files, output_folder, caption_text, font_path, font_size_divisor, text_width_ratio, text_color, stroke_color, stroke_width)