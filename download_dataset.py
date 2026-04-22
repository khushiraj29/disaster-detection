"""
download_dataset.py
-------------------
Comprehensive dataset builder for the Disaster Detection model.

Strategy:
  1. Try downloading real images from Wikimedia Commons (free, no API key)
  2. Supplement with programmatically generated synthetic training images
     using NumPy + PIL to ensure each class has enough images for training.

Each class will end up with at least 50 images for a viable training set.
"""

import os
import sys
import random
import urllib.request
import urllib.parse
import json
import concurrent.futures
import struct
import zlib
from io import BytesIO

# --- Configuration ---
OUTPUT_DIR = "data/raw"
MIN_IMAGES_PER_CLASS = 300  # Minimum images we want per class
WIKIMEDIA_TIMEOUT = 15
DOWNLOAD_TIMEOUT = 10

# Search queries for Wikimedia Commons (multiple queries per class)
WIKIMEDIA_QUERIES = {
    "earthquake_damage": [
        "earthquake damage building",
        "earthquake ruins",
        "collapsed building earthquake",
        "seismic damage",
    ],
    "fire": [
        "wildfire",
        "forest fire",
        "house fire",
        "building fire disaster",
    ],
    "flood": [
        "flood disaster",
        "flooded street",
        "river flood",
        "flooding city",
    ],
    "landslide": [
        "landslide",
        "mudslide disaster",
        "rockslide",
        "debris flow",
    ],
    "normal": [
        "city street",
        "residential neighborhood",
        "urban park",
        "suburban road",
        "village street",
    ],
}


def download_image(img_url, img_path):
    """Download a single image from a URL."""
    try:
        req = urllib.request.Request(
            img_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36"
            },
        )
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
            data = resp.read()
            if len(data) > 1000:  # Skip tiny/broken downloads
                with open(img_path, "wb") as f:
                    f.write(data)
                return True
    except Exception:
        pass
    return False


def download_from_wikimedia(class_name, queries, class_dir):
    """Download images from Wikimedia Commons for a given class."""
    all_urls = []

    for query in queries:
        api_url = (
            "https://commons.wikimedia.org/w/api.php?"
            "action=query&generator=search"
            f"&gsrsearch={urllib.parse.quote(query)}"
            "&gsrnamespace=6&gsrlimit=50"
            "&prop=imageinfo&iiprop=url&format=json"
        )
        try:
            req = urllib.request.Request(
                api_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (DisasterDetection/1.0; "
                                  "Academic Research) AppleWebKit/537.36"
                },
            )
            with urllib.request.urlopen(req, timeout=WIKIMEDIA_TIMEOUT) as resp:
                data = json.loads(resp.read().decode())
                pages = data.get("query", {}).get("pages", {})
                for page_info in pages.values():
                    if "imageinfo" in page_info:
                        url = page_info["imageinfo"][0]["url"]
                        ext = url.lower().split(".")[-1].split("?")[0]
                        if ext in ("jpg", "jpeg", "png"):
                            if url not in all_urls:
                                all_urls.append(url)
        except Exception as e:
            print(f"  [Wikimedia] Query '{query}' failed: {e}")

    # Download concurrently
    count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for i, url in enumerate(all_urls[:200]):
            path = os.path.join(class_dir, f"wiki_{i:03d}.jpg")
            futures[executor.submit(download_image, url, path)] = path

        for future in concurrent.futures.as_completed(futures):
            if future.result():
                count += 1

    return count


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Image Generation using pure Python (NumPy + struct for PNG)
# ─────────────────────────────────────────────────────────────────────────────

def _save_png(filepath, pixels):
    """Save a numpy array (H, W, 3) as a PNG file using pure Python."""
    h, w, _ = pixels.shape
    pixels = pixels.astype("uint8")

    def make_chunk(chunk_type, data):
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    # PNG signature
    sig = b"\x89PNG\r\n\x1a\n"

    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    ihdr = make_chunk(b"IHDR", ihdr_data)

    # IDAT
    raw_data = b""
    for y in range(h):
        raw_data += b"\x00"  # filter byte
        raw_data += pixels[y].tobytes()
    compressed = zlib.compress(raw_data, 6)
    idat = make_chunk(b"IDAT", compressed)

    # IEND
    iend = make_chunk(b"IEND", b"")

    with open(filepath, "wb") as f:
        f.write(sig + ihdr + idat + iend)


def _gradient(h, w, color_top, color_bottom):
    """Create a vertical gradient image."""
    img = np.zeros((h, w, 3), dtype=np.float64)
    for y in range(h):
        t = y / max(h - 1, 1)
        for c in range(3):
            img[y, :, c] = color_top[c] * (1 - t) + color_bottom[c] * t
    return img


def _add_noise(img, strength=15):
    """Add random noise to an image."""
    noise = np.random.normal(0, strength, img.shape)
    return np.clip(img + noise, 0, 255)


def _draw_rect(img, y1, x1, y2, x2, color, alpha=1.0):
    """Draw a filled rectangle on the image."""
    h, w = img.shape[:2]
    y1, y2 = max(0, int(y1)), min(h, int(y2))
    x1, x2 = max(0, int(x1)), min(w, int(x2))
    for c in range(3):
        img[y1:y2, x1:x2, c] = img[y1:y2, x1:x2, c] * (1 - alpha) + color[c] * alpha
    return img


def _draw_circle(img, cy, cx, radius, color, alpha=0.8):
    """Draw a filled circle on the image."""
    h, w = img.shape[:2]
    Y, X = np.ogrid[:h, :w]
    mask = ((X - cx) ** 2 + (Y - cy) ** 2) <= radius ** 2
    for c in range(3):
        img[:, :, c] = np.where(mask, img[:, :, c] * (1 - alpha) + color[c] * alpha, img[:, :, c])
    return img


def _draw_irregular_shape(img, points, color, alpha=0.7):
    """Draw an irregular polygon-ish blob."""
    h, w = img.shape[:2]
    Y, X = np.ogrid[:h, :w]
    # Use a simple approach: draw circles at each point and between them
    for i in range(len(points)):
        cy, cx = points[i]
        _draw_circle(img, cy, cx, random.randint(15, 40), color, alpha * 0.5)
    return img


import numpy as np

random.seed(42)
np.random.seed(42)


def generate_earthquake_image(size=224):
    """Generate a synthetic earthquake damage scene."""
    img = _gradient(size, size, 
                    [random.randint(140, 180)] * 3,  # gray sky
                    [random.randint(100, 140), random.randint(80, 110), random.randint(60, 80)])
    
    # Draw damaged/tilted buildings
    num_buildings = random.randint(3, 6)
    for _ in range(num_buildings):
        bw = random.randint(30, 70)
        bh = random.randint(60, 150)
        bx = random.randint(0, size - bw)
        by = size - bh - random.randint(0, 30)
        
        # Building color (concrete/brown tones)
        bc = [random.randint(100, 160), random.randint(90, 140), random.randint(70, 120)]
        _draw_rect(img, by, bx, by + bh, bx + bw, bc, 0.85)
        
        # Cracks (darker lines)
        for _ in range(random.randint(2, 5)):
            cx = bx + random.randint(5, bw - 5)
            cy = by + random.randint(5, bh - 5)
            _draw_rect(img, cy, cx, cy + random.randint(10, 30), cx + 2, [40, 35, 30], 0.9)
    
    # Debris on ground
    for _ in range(random.randint(8, 20)):
        dx = random.randint(0, size - 15)
        dy = random.randint(size - 60, size - 5)
        ds = random.randint(5, 20)
        dc = [random.randint(80, 140), random.randint(70, 120), random.randint(50, 100)]
        _draw_rect(img, dy, dx, dy + ds, dx + ds, dc, 0.7)
    
    return _add_noise(img, random.randint(8, 20)).astype(np.uint8)


def generate_fire_image(size=224):
    """Generate a synthetic fire/wildfire scene."""
    # Dark/smoky sky gradient
    img = _gradient(size, size,
                    [random.randint(40, 80), random.randint(20, 50), random.randint(10, 30)],
                    [random.randint(60, 100), random.randint(40, 60), random.randint(20, 40)])
    
    # Ground
    _draw_rect(img, size * 2 // 3, 0, size, size, 
               [random.randint(30, 60), random.randint(20, 40), random.randint(10, 25)], 0.7)
    
    # Fire elements (bright orange/yellow/red)
    num_flames = random.randint(4, 10)
    for _ in range(num_flames):
        fx = random.randint(10, size - 30)
        fy = random.randint(size // 4, size - 20)
        fr = random.randint(15, 50)
        
        # Outer flame (orange/red)
        _draw_circle(img, fy, fx, fr,
                     [random.randint(200, 255), random.randint(80, 160), random.randint(0, 40)], 0.7)
        # Inner flame (yellow/white)
        _draw_circle(img, fy + 5, fx, fr // 2,
                     [random.randint(240, 255), random.randint(200, 255), random.randint(50, 150)], 0.6)
    
    # Smoke clouds (gray/dark)
    for _ in range(random.randint(3, 7)):
        sx = random.randint(0, size)
        sy = random.randint(0, size // 2)
        sr = random.randint(20, 60)
        sg = random.randint(60, 120)
        _draw_circle(img, sy, sx, sr, [sg, sg - 10, sg - 20], 0.4)
    
    # Orange/red glow
    for y in range(size):
        for x in range(size):
            if y > size // 2:
                glow = max(0, 1 - abs(x - size // 2) / (size // 2)) * 0.15
                img[y, x, 0] = min(255, img[y, x, 0] + 60 * glow)
                img[y, x, 1] = min(255, img[y, x, 1] + 20 * glow)
    
    return _add_noise(img, random.randint(10, 25)).astype(np.uint8)


def generate_flood_image(size=224):
    """Generate a synthetic flood scene."""
    # Sky
    img = _gradient(size, size,
                    [random.randint(100, 160), random.randint(120, 180), random.randint(160, 220)],
                    [random.randint(80, 130), random.randint(100, 150), random.randint(140, 190)])
    
    # Water level (covers bottom portion)
    water_line = random.randint(size // 3, size // 2)
    water_color = [random.randint(60, 110), random.randint(90, 140), random.randint(120, 180)]
    _draw_rect(img, water_line, 0, size, size, water_color, 0.85)
    
    # Water reflections / ripples
    for _ in range(random.randint(15, 30)):
        ry = random.randint(water_line + 10, size - 5)
        rx = random.randint(0, size - 30)
        rw = random.randint(15, 60)
        lighter = [min(255, c + 30) for c in water_color]
        _draw_rect(img, ry, rx, ry + 2, rx + rw, lighter, 0.3)
    
    # Partially submerged buildings
    num_buildings = random.randint(2, 4)
    for _ in range(num_buildings):
        bw = random.randint(25, 55)
        bh = random.randint(40, 100)
        bx = random.randint(10, size - bw - 10)
        by = water_line - bh // 2  # Building partially in water
        bc = [random.randint(120, 180), random.randint(110, 160), random.randint(100, 140)]
        _draw_rect(img, by, bx, by + bh, bx + bw, bc, 0.75)
        
        # Roof
        _draw_rect(img, by - 5, bx - 3, by, bx + bw + 3,
                   [random.randint(80, 130), random.randint(40, 70), random.randint(30, 50)], 0.8)
    
    # Floating debris
    for _ in range(random.randint(5, 12)):
        dx = random.randint(0, size - 10)
        dy = random.randint(water_line + 5, size - 10)
        ds = random.randint(3, 12)
        _draw_rect(img, dy, dx, dy + ds, dx + ds + random.randint(0, 10),
                   [random.randint(80, 130), random.randint(60, 100), random.randint(40, 70)], 0.6)
    
    return _add_noise(img, random.randint(8, 18)).astype(np.uint8)


def generate_landslide_image(size=224):
    """Generate a synthetic landslide scene."""
    # Sky
    img = _gradient(size, size,
                    [random.randint(130, 180), random.randint(140, 190), random.randint(160, 210)],
                    [random.randint(100, 140), random.randint(80, 110), random.randint(50, 80)])
    
    # Mountain/hillside
    slope_start = random.randint(size // 4, size // 3)
    for x in range(size):
        slope_y = slope_start + int(30 * np.sin(x * 0.03)) + x // 4
        slope_y = min(slope_y, size - 1)
        green = [random.randint(40, 80), random.randint(80, 130), random.randint(30, 60)]
        _draw_rect(img, slope_y, x, size, x + 1, green, 0.7)
    
    # Landslide scar (brown/mud colored diagonal strip)
    scar_x = random.randint(size // 4, size // 2)
    scar_w = random.randint(40, 80)
    for y in range(slope_start, size):
        offset = (y - slope_start) * random.uniform(0.3, 0.7)
        sx1 = int(scar_x + offset)
        sx2 = int(sx1 + scar_w + random.randint(-10, 10))
        mud_color = [random.randint(120, 170), random.randint(80, 120), random.randint(40, 70)]
        _draw_rect(img, y, max(0, sx1), y + 1, min(size, sx2), mud_color, 0.8)
    
    # Rocks and debris at the bottom
    for _ in range(random.randint(10, 25)):
        rx = random.randint(scar_x, min(size - 10, scar_x + scar_w + 40))
        ry = random.randint(size - 60, size - 5)
        rs = random.randint(5, 20)
        rc = [random.randint(100, 150), random.randint(80, 120), random.randint(60, 90)]
        _draw_circle(img, ry, rx, rs, rc, 0.7)
    
    # Fallen trees (simple lines)
    for _ in range(random.randint(2, 5)):
        tx = random.randint(scar_x - 20, scar_x + scar_w)
        ty = random.randint(size // 2, size - 30)
        _draw_rect(img, ty, tx, ty + 3, tx + random.randint(20, 50),
                   [random.randint(60, 90), random.randint(40, 60), random.randint(20, 40)], 0.7)
    
    return _add_noise(img, random.randint(8, 18)).astype(np.uint8)


def generate_normal_image(size=224):
    """Generate a synthetic normal street/park scene."""
    # Blue sky
    img = _gradient(size, size,
                    [random.randint(100, 160), random.randint(150, 210), random.randint(220, 255)],
                    [random.randint(160, 210), random.randint(200, 240), random.randint(220, 255)])
    
    # Ground (road or grass)
    ground_y = random.randint(size // 2, size * 2 // 3)
    is_road = random.random() > 0.4
    if is_road:
        ground_color = [random.randint(80, 120), random.randint(80, 110), random.randint(80, 110)]
    else:
        ground_color = [random.randint(50, 90), random.randint(120, 170), random.randint(40, 70)]
    _draw_rect(img, ground_y, 0, size, size, ground_color, 0.85)
    
    # Buildings
    num_buildings = random.randint(2, 5)
    for _ in range(num_buildings):
        bw = random.randint(25, 60)
        bh = random.randint(50, ground_y - 20)
        bx = random.randint(0, size - bw)
        by = ground_y - bh
        # Colorful buildings
        bc = [random.randint(150, 240), random.randint(150, 240), random.randint(150, 240)]
        _draw_rect(img, by, bx, ground_y, bx + bw, bc, 0.8)
        
        # Windows
        for wy in range(by + 8, ground_y - 8, 15):
            for wx in range(bx + 5, bx + bw - 5, 12):
                _draw_rect(img, wy, wx, wy + 8, wx + 7,
                           [random.randint(150, 200), random.randint(200, 240), random.randint(230, 255)], 0.7)
    
    # Trees
    for _ in range(random.randint(1, 4)):
        tx = random.randint(10, size - 30)
        # Trunk
        _draw_rect(img, ground_y - 40, tx, ground_y, tx + 6,
                   [random.randint(80, 120), random.randint(50, 70), random.randint(20, 40)], 0.8)
        # Canopy
        _draw_circle(img, ground_y - 55, tx + 3, random.randint(15, 30),
                     [random.randint(30, 70), random.randint(120, 180), random.randint(30, 70)], 0.7)
    
    # Sun or clouds
    if random.random() > 0.5:
        _draw_circle(img, random.randint(20, 50), random.randint(size - 60, size - 20),
                     random.randint(15, 25), [255, 240, 180], 0.6)
    
    return _add_noise(img, random.randint(5, 12)).astype(np.uint8)


GENERATORS = {
    "earthquake_damage": generate_earthquake_image,
    "fire": generate_fire_image,
    "flood": generate_flood_image,
    "landslide": generate_landslide_image,
    "normal": generate_normal_image,
}


def count_existing_images(class_dir):
    """Count valid image files in a directory."""
    if not os.path.exists(class_dir):
        return 0
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(
        1 for f in os.listdir(class_dir)
        if os.path.splitext(f)[1].lower() in extensions
        and os.path.getsize(os.path.join(class_dir, f)) > 1000
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("  Disaster Detection — Dataset Builder")
    print("=" * 60)
    
    total_downloaded = 0
    total_generated  = 0
    
    for class_name, queries in WIKIMEDIA_QUERIES.items():
        class_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        existing = count_existing_images(class_dir)
        print(f"\n[{class_name}] Existing images: {existing}")
        
        # Step 1: Try downloading from Wikimedia Commons
        if existing < MIN_IMAGES_PER_CLASS:
            print(f"  → Downloading from Wikimedia Commons...")
            downloaded = download_from_wikimedia(class_name, queries, class_dir)
            total_downloaded += downloaded
            existing = count_existing_images(class_dir)
            print(f"  → Downloaded {downloaded} images (total now: {existing})")
        
        # Step 2: Generate synthetic images to reach minimum
        needed = max(0, MIN_IMAGES_PER_CLASS - existing)
        if needed > 0:
            print(f"  → Generating {needed} synthetic images...")
            gen_func = GENERATORS[class_name]
            for i in range(needed):
                # Each synthetic image is unique due to random parameters
                np.random.seed(existing + i + hash(class_name) % 10000)
                random.seed(existing + i + hash(class_name) % 10000)
                
                img = gen_func(224)
                # Vary brightness/contrast for more diversity
                factor = random.uniform(0.8, 1.2)
                img = np.clip(img * factor, 0, 255).astype(np.uint8)
                
                filepath = os.path.join(class_dir, f"synth_{i:03d}.png")
                _save_png(filepath, img)
            
            total_generated += needed
            existing = count_existing_images(class_dir)
            print(f"  → Generated {needed} synthetic images (total now: {existing})")
    
    print(f"\n{'=' * 60}")
    print(f"  Dataset Build Complete!")
    print(f"  Downloaded: {total_downloaded} images from Wikimedia")
    print(f"  Generated:  {total_generated} synthetic images")
    print(f"{'=' * 60}")
    
    # Print final summary
    print(f"\nFinal dataset summary:")
    for class_name in WIKIMEDIA_QUERIES:
        class_dir = os.path.join(OUTPUT_DIR, class_name)
        count = count_existing_images(class_dir)
        print(f"  {class_name:20s}: {count} images")
    
    total = sum(
        count_existing_images(os.path.join(OUTPUT_DIR, c))
        for c in WIKIMEDIA_QUERIES
    )
    print(f"  {'TOTAL':20s}: {total} images")
    print(f"\nDataset ready at: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
