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

def resize_and_crop(image, target_width, target_height):
    """
    Resizes an image to a target resolution by cropping and scaling.
    """
    original_width, original_height = image.size
    target_aspect = target_width / target_height
    original_aspect = original_width / original_height

    if original_aspect > target_aspect:
        # Original image is wider than target, crop width
        new_width = int(target_aspect * original_height)
        left = (original_width - new_width) / 2
        top = 0
        right = left + new_width
        bottom = original_height
        image = image.crop((left, top, right, bottom))
    elif original_aspect < target_aspect:
        # Original image is taller than target, crop height
        new_height = int(original_width / target_aspect)
        left = 0
        top = (original_height - new_height) / 2
        right = original_width
        bottom = top + new_height
        image = image.crop((left, top, right, bottom))

    # Resize the cropped image to the target resolution
    return image.resize((target_width, target_height), Image.LANCZOS)

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

def process_images(image_paths, output_folder, caption_text, font_path, font_size_divisor, text_width_ratio, text_color, stroke_color, stroke_width, placement_mode="Automatic", manual_placements=None, resolution=None):
    """
    Processes a list of images to add a caption based on the selected placement mode.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize face detector only if needed
    face_detector = None
    if placement_mode == "Automatic":
        mp_face_detection = mp.solutions.face_detection
        face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    safe_print(f"Starting to process {len(image_paths)} images with mode: {placement_mode}...")

    # --- Pre-determine the single manual position if that mode is selected ---
    manual_pos_for_all = None
    if placement_mode == "Manual - Same for All" and manual_placements:
        # Get the first available placement
        manual_pos_for_all = next(iter(manual_placements.values()), None)

    for image_path in tqdm(image_paths, desc="Captioning Images", disable=not sys.stdout):
        filename = os.path.basename(image_path)
        base_image = Image.open(image_path).convert("RGBA")

        if resolution:
            base_image = resize_and_crop(base_image, resolution[0], resolution[1])

        draw = ImageDraw.Draw(base_image)
        original_width, original_height = base_image.size

        # --- CAPTION STYLING LOGIC (same for all modes) ---
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

        # --- PLACEMENT LOGIC ---
        current_y = 0
        x_coords = []

        if placement_mode == "Automatic":
            # --- AUTOMATIC PLACEMENT LOGIC ---
            cv_image = cv2.imread(image_path)
            
            def rotate_image(image, angle):
                h, w = image.shape[:2]
                center = (w / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
                nW, nH = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
                M[0, 2] += (nW / 2) - center[0]
                M[1, 2] += (nH / 2) - center[1]
                return cv2.warpAffine(image, M, (nW, nH)), M

            def get_face_detections_with_rotation(cv_img):
                for angle in [0, 15, -15, 30, -30]:
                    img_to_proc, M = (cv_img, None) if angle == 0 else rotate_image(cv_img, angle)
                    results = face_detector.process(cv2.cvtColor(img_to_proc, cv2.COLOR_BGR2RGB))
                    if results.detections:
                        return results, M, img_to_proc.shape[1], img_to_proc.shape[0]
                return None, None, None, None

            results, M, processed_width, processed_height = get_face_detections_with_rotation(cv_image)
            
            occupied_zones = []
            if results:
                inv_M = cv2.invertAffineTransform(M) if M is not None else None
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x, y, w, h = int(bbox.xmin * processed_width), int(bbox.ymin * processed_height), int(bbox.width * processed_width), int(bbox.height * processed_height)
                    corners = cv2.transform(np.array([[[x, y], [x+w, y], [x+w, y+h], [x, y+h]]], dtype=np.float32), inv_M) if M is not None else np.array([[[x, y], [x+w, y], [x+w, y+h], [x, y+h]]])
                    y_coords = corners[0, :, 1]
                    occupied_zones.append((int(np.min(y_coords)), int(np.max(y_coords))))

            if occupied_zones:
                occupied_zones.sort()
                merged = [occupied_zones[0]]
                for start, end in occupied_zones[1:]:
                    last_start, last_end = merged[-1]
                    if start <= last_end:
                        merged[-1] = (last_start, max(last_end, end))
                    else:
                        merged.append((start, end))
                occupied_zones = merged

            safe_zones = []
            last_y = 0
            for start, end in occupied_zones:
                safe_zones.append((last_y, start))
                last_y = end
            safe_zones.append((last_y, original_height))

            best_zone = max(safe_zones, key=lambda z: z[1] - z[0])
            zone_start, zone_end = best_zone
            zone_height = zone_end - zone_start
            
            if zone_height >= block_height:
                offset = (zone_height - block_height) / 2
                current_y = zone_start + offset
            else:
                margin = int(original_height * 0.05)
                current_y = original_height - block_height - margin

        elif placement_mode in ["Manual - Same for All", "Manual - Individual"]:
            # --- MANUAL PLACEMENT LOGIC ---
            pos = manual_pos_for_all if placement_mode == "Manual - Same for All" else manual_placements.get(image_path)
            
            if pos:
                manual_x, manual_y = pos
                # In manual mode, the click point is the center of the text block
                current_y = manual_y - (block_height / 2)
                # We'll calculate x for each line to center it on manual_x
                for line in lines:
                    line_bbox = draw.textbbox((0, 0), line, font=font)
                    line_width = line_bbox[2] - line_bbox[0]
                    x_coords.append(manual_x - (line_width / 2))
            else:
                # Fallback for safety, though GUI should prevent this
                margin = int(original_height * 0.05)
                current_y = original_height - block_height - margin


        # --- DRAWING LOGIC ---
        # For a more authentic "TikTok" feel, we'll draw a soft shadow first.
        shadow_offset = max(1, int(stroke_width * 1.5)) # Make shadow slightly larger than the stroke
        shadow_color = (0, 0, 0, 150) # Semi-transparent black for a softer shadow

        for i, line in enumerate(lines):
            # For automatic, calculate x centered for each line
            if not x_coords:
                line_bbox = draw.textbbox((0, 0), line, font=font)
                line_width = line_bbox[2] - line_bbox[0]
                x = (original_width - line_width) / 2
            else:
                # For manual, use pre-calculated x
                x = x_coords[i]

            # 1. Draw the soft shadow by drawing the text at several offset positions
            draw.text((x - shadow_offset, current_y), line, font=font, fill=shadow_color)
            draw.text((x + shadow_offset, current_y), line, font=font, fill=shadow_color)
            draw.text((x, current_y - shadow_offset), line, font=font, fill=shadow_color)
            draw.text((x, current_y + shadow_offset), line, font=font, fill=shadow_color)
            
            # 2. Draw the main text with its original crisp stroke on top
            draw.text((x, current_y), line, font=font, fill=text_color, stroke_width=stroke_width, stroke_fill=stroke_color)
            
            current_y += line_height

        # --- SAVE IMAGE ---
        output_path = os.path.join(output_folder, filename)
        if filename.lower().endswith(('.jpg', '.jpeg')):
            base_image = base_image.convert("RGB")
        base_image.save(output_path)

    safe_print(f"✅ Success! All images have been captioned and saved in the \"{output_folder}\" folder.")