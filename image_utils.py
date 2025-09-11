from PIL import Image, ImageOps
def safe_image_loader(path: str):
    max_side = 4096 
    try:
        with Image.open(path) as im:
        
            try:
                im = ImageOps.exif_transpose(im)
            except Exception:
                pass
            mode = im.mode
            if mode in ("P", "LA", "RGBA"):
            
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
    if img.width * img.height > 20_000_000:
        img = img.copy()
        img.thumbnail((2048, 2048)) 
    return img