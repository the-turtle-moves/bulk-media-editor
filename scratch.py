import os
import shutil
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import textwrap
import cv2
import mediapipe as mp # New SOTA library for face detection

# --- 1. SET YOUR CONFIGURATION HERE ---

# The folder containing the original images
input_folder = 'IMG_7280'

# The folder where captioned images will be saved
output_folder = 'captioned_images'

# The text you want to write on the images
caption_text = "Sample Caption Text, this will be centered on the image. Adjust as needed."

# Font settings
font_path = 'Roboto-VariableFont_wdth,wght.ttf'
font_size_divisor = 10
text_width_ratio = 0.9
text_color = (255, 255, 255)
stroke_color = (0, 0, 0)
stroke_width = 2


# --- 2. SCRIPT LOGIC (No need to edit below) ---

# Initialize the SOTA MediaPipe Face Detector
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

print(f"Starting to process {len(image_files)} images with MediaPipe SOTA face detection...")

for filename in tqdm(image_files, desc="Captioning Images"):
    image_path = os.path.join(input_folder, filename)

    # --- NEW: SOTA FACE DETECTION LOGIC ---
    # 1. Load image and prepare it for MediaPipe
    cv_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) # MediaPipe requires RGB
    results = face_detector.process(rgb_image)

    # 2. Extract face bounding boxes in (x, y, w, h) format
    faces = []
    if results.detections:
        img_height, img_width, _ = cv_image.shape
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = int(bboxC.xmin * img_width), int(bboxC.ymin * img_height), \
                         int(bboxC.width * img_width), int(bboxC.height * img_height)
            faces.append((x, y, w, h))
            
    # 3. Decide placement (this logic remains the same)
    placement = 'bottom'
    image_height_cv = cv_image.shape[0]
    for (x, y, w, h) in faces:
        if y + h > image_height_cv / 2:
            placement = 'top'
            break

    # --- CAPTIONING LOGIC (Now with SOTA face-awareness) ---
    with Image.open(image_path).convert("RGBA") as base_image:
        draw = ImageDraw.Draw(base_image)
        image_width, image_height_pil = base_image.size
        
        font_size = int(image_width / font_size_divisor)
        font = ImageFont.truetype(font_path, font_size)

        single_line_bbox = draw.textbbox((0, 0), caption_text, font=font)
        single_line_width = single_line_bbox[2] - single_line_bbox[0]
        line_height = single_line_bbox[3] - single_line_bbox[1]
        
        max_text_width = image_width * text_width_ratio
        lines = [caption_text]
        if single_line_width > max_text_width:
            avg_char_width = single_line_width / len(caption_text)
            wrap_at_chars = int(max_text_width / avg_char_width)
            lines = textwrap.wrap(caption_text, width=wrap_at_chars)

        block_height = len(lines) * line_height
        margin = int(image_height_pil * 0.05)
        
        if placement == 'bottom':
            current_y = image_height_pil - block_height - margin
        else: # 'top'
            current_y = margin

        for line in lines:
            line_bbox = draw.textbbox((0, 0), line, font=font)
            line_width = line_bbox[2] - line_bbox[0]
            x = (image_width - line_width) / 2
            draw.text((x, current_y), line, font=font, fill=text_color, stroke_width=stroke_width, stroke_fill=stroke_color)
            current_y += line_height

        output_path = os.path.join(output_folder, filename)
        if filename.lower().endswith(('.jpg', '.jpeg')):
            base_image = base_image.convert("RGB")
        base_image.save(output_path)

print(f"\n✅ Success! All images have been captioned and saved in the '{output_folder}' folder.")