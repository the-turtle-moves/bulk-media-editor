import os
import sys
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import textwrap
import cv2
import mediapipe as mp
import numpy as np
import random

def replace_quotes(text):
    text = str(text)
    in_double = False
    in_single = False
    res = ''
    for char in text:
        if char == '"':
            res += '“' if not in_double else '”'
            in_double = not in_double
        elif char == "'":
            res += '‘' if not in_single else '’'
            in_single = not in_single
        else:
            res += char
    return res

def wrap_text(caption_text, font_path, font_size, text_width_ratio, image_width):
    """
    Wraps the text based on the image width and returns a newline-separated string.
    """
    dummy_draw = ImageDraw.Draw(Image.new('RGBA', (0,0)))
    # Scale font size based on image width for an initial good wrapping estimate.
    scaled_font_size = int(font_size * (image_width / 1080))
    font = ImageFont.truetype(font_path, scaled_font_size)
    
    single_line_bbox = dummy_draw.textbbox((0, 0), caption_text, font=font)
    single_line_width = single_line_bbox[2] - single_line_bbox[0]
    
    max_text_width = image_width * text_width_ratio
    lines = [caption_text]
    if single_line_width > max_text_width:
        avg_char_width = single_line_width / len(caption_text) if len(caption_text) > 0 else 1
        wrap_at_chars = int(max_text_width / avg_char_width) if avg_char_width > 0 else 20
        lines = textwrap.wrap(caption_text, width=wrap_at_chars)
    
    return "\n".join(lines)

def draw_caption(draw, caption_text, font_path, font_size, text_width_ratio, text_color, stroke_color, stroke_width, original_width, original_height, target_caption_width=None):
    # Scale font size based on image width. The base font_size is for a 1080px wide image.
    scaled_font_size = int(font_size * (original_width / 1080))
    font = ImageFont.truetype(font_path, scaled_font_size)
    
    single_line_bbox = draw.textbbox((0, 0), caption_text, font=font)
    single_line_width = single_line_bbox[2] - single_line_bbox[0]
    line_height = single_line_bbox[3] - single_line_bbox[1]
    
    max_text_width = target_caption_width if target_caption_width is not None else (original_width * text_width_ratio)
    lines = [caption_text]
    if single_line_width > max_text_width:
        avg_char_width = single_line_width / len(caption_text)
        wrap_at_chars = int(max_text_width / avg_char_width) if avg_char_width > 0 else 20
        lines = textwrap.wrap(caption_text, width=wrap_at_chars)
    
    block_height = len(lines) * line_height

    return lines, block_height, font, scaled_font_size

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

def multiline_bbox(draw, text, font, stroke_w=0, **kw):
    """Return (w, h) for multiline text across Pillow versions."""
    if hasattr(draw, "multiline_textbbox"):          # Pillow >= 8.0
        l, t, r, b = draw.multiline_textbbox(
            (0, 0), text, font=font,
            stroke_width=stroke_w, **kw
        )
        return r - l, b - t
    else:                                            # legacy fallback
        return draw.multiline_textsize(
            text, font=font, stroke_width=stroke_w, **kw
        )

def get_automatic_placement_coords(image_input, original_width, original_height, block_height, face_detector):
    if isinstance(image_input, str):
        cv_image = cv2.imread(image_input)
    else: # Assumes PIL Image
        cv_image = cv2.cvtColor(np.array(image_input.convert('RGB')), cv2.COLOR_RGB2BGR)
    
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
        for angle in [0, 20, -20, 40, -40, 60, -60, 90, -90]:
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

    return current_y

def generate_captioned_image(base_image, settings, config, face_detector, random_tilt=False):
    original_width, original_height = base_image.size
    caption_text = settings.get('caption', '')
    wrapped_caption = settings.get('wrapped_caption', '')

    if not caption_text:
        return base_image, None

    # Use the pre-wrapped caption for drawing, but the original for quotes
    final_caption_text = replace_quotes(wrapped_caption if wrapped_caption else caption_text)
    
    scale_x = settings.get('scale_x', 1.0)
    scale_y = settings.get('scale_y', 1.0)

    # --- CAPTION STYLING LOGIC ---
    avg_scale = (scale_x + scale_y) / 2
    # The base font_size is scaled by the user's resize operations and the image's own width.
    scaled_font_size = int(config['font_size'] * (original_width / 1080) * avg_scale)
    font = ImageFont.truetype(resource_path(config['font_path']), scaled_font_size)
    scaled_stroke_width = int(config['stroke_width'] * avg_scale)

    dummy_draw = ImageDraw.Draw(Image.new('RGBA', (0,0)))
    
    # Get the bounding box of the final wrapped text
    tw, th = multiline_bbox(dummy_draw, final_caption_text, font, stroke_w=scaled_stroke_width, spacing=12)
    th += int(scaled_font_size * 0.2) # Add 20% padding to prevent clipping
    
    # --- PLACEMENT LOGIC ---
    x = settings.get('x')
    y = settings.get('y')
    
    # Block height for automatic placement needs to be calculated based on the unscaled font size
    if y is None:
        initial_font_size = int(config['font_size'] * (original_width / 1080))
        initial_font = ImageFont.truetype(resource_path(config['font_path']), initial_font_size)
        _, initial_th = multiline_bbox(dummy_draw, final_caption_text, font=initial_font, spacing=12)
        y = get_automatic_placement_coords(base_image, original_width, original_height, initial_th, face_detector)

    if x is None:
        x = (original_width - tw) / 2

    # --- TILT LOGIC ---
    angle = 0
    if random_tilt:
        if 'tilt_angle' not in settings or settings['tilt_angle'] == 0:
            settings['tilt_angle'] = random.uniform(-30, 30)
        angle = settings['tilt_angle']
    else:
        settings['tilt_angle'] = 0

    # --- DRAWING LOGIC ---
    text_layer = Image.new('RGBA', (tw, th), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_layer)
    text_draw.multiline_text(
        (0, 0), final_caption_text, font=font, fill=tuple(config['text_color']),
        stroke_width=scaled_stroke_width, stroke_fill=tuple(config['stroke_color']),
        spacing=12, align="center"
    )

    if angle != 0:
        text_layer = text_layer.rotate(angle, expand=True, resample=Image.BICUBIC)
        new_tw, new_th = text_layer.size
        x -= (new_tw - tw) / 2
        y -= (new_th - th) / 2

    base_image.paste(text_layer, (int(x), int(y)), text_layer)
    return base_image, (x, y, tw, th)

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

def process_images(image_paths, output_folder, font_path, font_size, text_width_ratio, text_color, stroke_color, stroke_width, resolution=None, image_settings=None, progress_callback=None, random_tilt=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if image_settings is None:
        image_settings = {}

    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    # This config object is a bit redundant, but it helps to pass all the settings to the new function
    config = {
        'font_path': font_path,
        'font_size': font_size,
        'text_width_ratio': text_width_ratio,
        'text_color': text_color,
        'stroke_color': stroke_color,
        'stroke_width': stroke_width
    }

    safe_print(f"Starting to process {len(image_paths)} images...")

    num_images = len(image_paths)
    for i, image_path in enumerate(image_paths):
        filename = os.path.basename(image_path)
        base_image = Image.open(image_path).convert("RGBA")

        if resolution:
            base_image = resize_and_crop(base_image, resolution[0], resolution[1])

        settings = image_settings.get(image_path, {})
        
        final_image, _ = generate_captioned_image(base_image, settings, config, face_detector, random_tilt)

        # --- SAVE IMAGE ---
        output_path = os.path.join(output_folder, filename)
        if filename.lower().endswith(('.jpg', '.jpeg')):
            final_image = final_image.convert("RGB")
        final_image.save(output_path)

        if progress_callback:
            progress_callback(i + 1, num_images)

    safe_print(f"✅ Success! All images have been captioned and saved in the \"{output_folder}\" folder.")