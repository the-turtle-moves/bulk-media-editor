# Bulk Media Editor

A desktop application for adding captions to images and videos.

## Features

*   Add text captions to images and videos.
*   Customize font, size, color, and outline of the text.
*   Automatic placement of captions to avoid faces in the media.
*   Manual placement and scaling of captions.
*   Process media in batches.
*   Group media and apply different captions to each group.
*   Resize images to a custom resolution.
*   Preview captions on the media before processing.
*   Supports various media formats including JPG, PNG, MP4, and MOV.

## Requirements

The application is built with Python and requires the following libraries:

*   Pillow
*   tqdm
*   opencv-python
*   mediapipe
*   numpy
*   moviepy

You can install them using pip:

```bash
pip install -r requirements.txt
```

## How to Run

To run the application from the source code, execute the following command:

```bash
python gui.py
```

## How to Build

A build script `build.bat` is provided to create a standalone executable using PyInstaller.

```bash
./build.bat
```

The executable will be located in the `dist` directory.

## Configuration

The `config.json` file allows you to customize the default settings for the captions.

| Setting             | Description                                           |
| ------------------- | ----------------------------------------------------- |
| `output_folder`     | The default folder where the captioned media is saved. |
| `font_path`         | The path to the font file (.otf or .ttf).             |
| `font_size`         | The size of the font.                                 |
| `text_width_ratio`  | The ratio of the text width to the image width.       |
| `text_color`        | The color of the text in RGB format.                  |
| `stroke_color`      | The color of the text outline in RGB format.          |
| `stroke_width`      | The width of the text outline.                        |
