"""Utility helpers for robust image loading & preprocessing."""
from PIL import Image, ImageOps, ImageFile
import warnings

Image.MAX_IMAGE_PIXELS = 300_000_000  # safety cap for huge images
ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes"
)

def safe_image_loader(path: str):
    """Open an image path robustly and return an RGB PIL.Image.

    Steps:
      - Handles truncated images (global flag set)
      - Applies EXIF orientation
      - Converts palette / alpha images to RGB composited on white
      - Soft downscales images whose max dimension exceeds max_side
      - Returns a blank white 256x256 image on failure
    """
    max_side = 4096
    try:
        with Image.open(path) as im:
            try:
                im = ImageOps.exif_transpose(im)
            except Exception:
                pass
            if im.mode in ("P", "LA", "RGBA"):
                im = im.convert("RGBA")
                bg = Image.new("RGB", im.size, (255, 255, 255))
                alpha = im.split()[-1]
                bg.paste(im, mask=alpha)
                im = bg
            else:
                im = im.convert("RGB")
            if max(im.size) > max_side:
                im.thumbnail((max_side, max_side))
            return im
    except Exception as e:
        print(f"[WARN] safe_image_loader failed for {path}: {e}")
        return Image.new("RGB", (256, 256), (255, 255, 255))


def safe_downscale(img):
    """Downscale very large images (>20 MP) in-place copy to 2048x2048 max."""
    try:
        if img.width * img.height > 20_000_000:
            img = img.copy()
            img.thumbnail((2048, 2048))
    except Exception:
        pass
    return img

__all__ = ["safe_image_loader", "safe_downscale"]