"""Render text in the typographic style used for IH-image and IH-mixed."""

import argparse
import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT_PATH = Path(__file__).resolve().parent / "assets" / "NotoSansCJK-Regular.ttc"


def render_text(text: str, output_path=None) -> Image.Image:
    """Return an RGB image; optionally save it to ``output_path``.

    Use 24 px text with word wrapping on a white background, capped at
    1,638,400 pixels.
    """
    font = ImageFont.truetype(str(FONT_PATH), 24)
    draw_kwargs = {
        "xy": (40, 40),
        "text": textwrap.fill(text, width=60 if len(text) < 1000 else 90),
        "spacing": 12,
        "font": font,
    }
    bounds = ImageDraw.Draw(Image.new("RGB", (0, 0))).textbbox(**draw_kwargs)
    width, height = max(800, bounds[2] + 80), bounds[3] + 100
    image = Image.new("RGB", (width, height), "#FFFFFF")
    ImageDraw.Draw(image).text(**draw_kwargs, fill="#000000")

    if width * height > 1_638_400:
        scale = math.sqrt(1_638_400 / (width * height))
        image = image.resize(
            (int(width * scale), int(height * scale)), Image.Resampling.LANCZOS
        )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Text to render; quote it in your shell.")
    parser.add_argument("--output", "-o", type=Path, default=Path("prompt.png"))
    args = parser.parse_args()
    render_text(args.text, args.output)
    print(args.output)
