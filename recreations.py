#!/usr/bin/env python3
"""
recreations.py — side-by-side composite images from recreation photo pairs.

Pairs are detected by matching filename prefixes before the trailing date:
  "Kiel - 20060116.jpg" + "Kiel - 20190428.jpg"  ->  "Kiel - composite.jpg"
  "1 - 2004-12-23 13.30.00.jpg" + "1 - 2021-12-25 18.19.46.heic"  ->  "1 - composite.jpg"

Usage:
  python recreations.py [FOLDER] [-o OUTPUT_FOLDER]

FOLDER defaults to the current directory.
Composites are saved to FOLDER/composites/ by default.

Requirements:
  pip install -r requirements.txt
  brew install exiftool
"""

import argparse, json, os, re, subprocess, sys, tempfile, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: pip install requests")

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("Missing dependency: pip install Pillow")

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("Missing dependency: pip install opencv-python numpy")

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    _HEIF_NATIVE = True
except ImportError:
    _HEIF_NATIVE = False  # falls back to sips on macOS


# ── Layout ────────────────────────────────────────────────────────────────────

TARGET_HEIGHT = 1800   # both photos scaled to this height in the composite
GAP_PX        = 30     # dark gap between the two photos
FALLBACK_AR   = 3 / 4  # w/h aspect ratio for center-crop when no face is detected

FONT_DATE_SZ  = 64
FONT_LOC_SZ   = 44
LABEL_PAD     = 18
LABEL_ALPHA   = 180    # 0-255 opacity of the dark label strip

FACE_MIN_RATIO = 0.05  # face_h / image_h — below this treated as noise/false positive


# ── EXIF ──────────────────────────────────────────────────────────────────────

def get_exif(path: Path) -> dict:
    r = subprocess.run(
        ['exiftool', '-j', '-n',  # -n: numeric output (decimal GPS, not DMS strings)
         '-DateTimeOriginal',
         '-GPSLatitude', '-GPSLongitude', '-GPSLatitudeRef', '-GPSLongitudeRef',
         str(path)],
        capture_output=True, text=True
    )
    if r.returncode != 0 or not r.stdout.strip():
        return {}
    data = json.loads(r.stdout)
    return data[0] if data else {}


def format_date(exif: dict) -> tuple[str, str]:
    """
    Returns (label_str, year_str) from EXIF DateTimeOriginal.
    label_str: "YYYY-MM-D HH.MM.SS" (non-padded day, matching filename convention)
    year_str:  "YYYY"
    """
    raw = exif.get('DateTimeOriginal', '')
    if not raw:
        return '', ''
    try:
        dt = datetime.strptime(raw, '%Y:%m:%d %H:%M:%S')
        label = f"{dt.year}-{dt.month:02d}-{dt.day} {dt.hour:02d}.{dt.minute:02d}.{dt.second:02d}"
        return label, str(dt.year)
    except ValueError:
        return raw, raw[:4]


def to_decimal_gps(exif: dict) -> tuple[float, float] | None:
    lat = exif.get('GPSLatitude')
    lon = exif.get('GPSLongitude')
    if lat is None or lon is None:
        return None
    # exiftool -n returns signed decimal degrees directly; no Ref adjustment needed
    return float(lat), float(lon)


# ── Geocoding ─────────────────────────────────────────────────────────────────

_geocache: dict[str, str] = {}


def reverse_geocode(lat: float, lon: float) -> str:
    key = f"{lat:.3f},{lon:.3f}"
    if key in _geocache:
        return _geocache[key]
    try:
        r = requests.get(
            'https://nominatim.openstreetmap.org/reverse',
            params={'lat': lat, 'lon': lon, 'format': 'json', 'zoom': 10,
                    'accept-language': 'en'},
            headers={'User-Agent': 'recreations-script/1.0'},
            timeout=10,
        )
        r.raise_for_status()
        addr = r.json().get('address', {})
        city = (addr.get('city') or addr.get('town') or
                addr.get('village') or addr.get('municipality') or '')
        country = addr.get('country', '')
        result = ', '.join(filter(None, [city, country]))
    except Exception as e:
        print(f"    geocoding failed: {e}")
        result = f"{lat:.2f}N, {lon:.2f}E"
    _geocache[key] = result
    time.sleep(1.1)  # Nominatim rate limit: 1 req/s
    return result


# ── Pairing ───────────────────────────────────────────────────────────────────

_IMAGE_EXTS = {'.jpg', '.jpeg', '.heic', '.png', '.tiff', '.tif'}

# Trailing date patterns: "20031227", "2004-12-23 13.30.00", "2026-07-11 12.48.33.11"
_DATE_SUFFIX = re.compile(
    r'[\s\-]+'
    r'(\d{4}-\d{1,2}-\d{1,2}(?:\s+\d{2}\.\d{2}\.\d{2}(?:\.\d+)?)?|\d{8})'
    r'\s*$'
)


def extract_prefix(stem: str) -> str:
    m = _DATE_SUFFIX.search(stem)
    return stem[:m.start()].strip(' -') if m else stem


def find_pairs(folder: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in _IMAGE_EXTS and not f.name.startswith('.'):
            groups[extract_prefix(f.stem)].append(f)
    return {k: v for k, v in groups.items() if len(v) == 2}


# ── Image loading ─────────────────────────────────────────────────────────────

def open_image(path: Path) -> Image.Image:
    from PIL import ImageOps
    if path.suffix.lower() == '.heic' and not _HEIF_NATIVE:
        # sips fallback (macOS built-in)
        tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        tmp.close()
        try:
            subprocess.run(
                ['sips', '-s', 'format', 'jpeg', str(path), '--out', tmp.name],
                capture_output=True, check=True
            )
            img = Image.open(tmp.name)
            img = ImageOps.exif_transpose(img)
            img = img.convert('RGB')
            img.load()
        finally:
            os.unlink(tmp.name)
        return img
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # apply EXIF rotation/flip
    img = img.convert('RGB')
    img.load()
    return img


# ── Face detection + zoom matching ───────────────────────────────────────────

_FRONTAL = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
_PROFILE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')


def detect_face(img: Image.Image) -> tuple[int, int, int, int] | None:
    """
    Returns (x, y, w, h) where w/h are from the LARGEST face (for zoom ratio),
    but x/y are set so the center lands on the GROUP centroid (for crop positioning).
    False positives smaller than FACE_MIN_RATIO are discarded.
    """
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    min_h = img.height * FACE_MIN_RATIO
    for cc in (_FRONTAL, _PROFILE):
        raw = cc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if not len(raw):
            continue
        faces = [f for f in raw if f[3] >= min_h]
        if not faces:
            continue
        largest = max(faces, key=lambda f: f[2] * f[3])
        x1 = min(int(f[0]) for f in faces)
        y1 = min(int(f[1]) for f in faces)
        x2 = max(int(f[0] + f[2]) for f in faces)
        y2 = max(int(f[1] + f[3]) for f in faces)
        gcx, gcy = (x1 + x2) // 2, (y1 + y2) // 2
        lw, lh = int(largest[2]), int(largest[3])
        return gcx - lw // 2, gcy - lh // 2, lw, lh
    return None


def _smart_crop(img: Image.Image, face: tuple | None,
                target_face_ratio: float | None, aspect: float
                ) -> tuple[Image.Image, float | None]:
    """
    Crop img so face occupies target_face_ratio of the frame height,
    centered on the face. Falls back to center crop when face is None.
    Returns (cropped_image, face_top_fraction) where face_top_fraction is
    the face's top edge as a fraction of the cropped image height (None if no face).
    """
    W, H = img.size

    if face is not None and target_face_ratio is not None:
        fx, fy, fw, fh = face
        cx = fx + fw // 2
        cy = fy + fh // 2
        crop_h = min(int(fh / target_face_ratio), H)
    else:
        fx = fy = None
        cx, cy = W // 2, H // 2
        crop_h = H

    crop_w = int(crop_h * aspect)
    if crop_w > W:
        crop_w = W
        crop_h = int(crop_w / aspect)

    left = max(0, min(cx - crop_w // 2, W - crop_w))
    top  = max(0, min(cy - crop_h // 2, H - crop_h))
    cropped = img.crop((left, top, left + crop_w, top + crop_h))

    face_top_frac = max(0.0, min(1.0, (fy - top) / crop_h)) if fy is not None else None
    return cropped, face_top_frac


def match_zoom(left: Image.Image, right: Image.Image,
               ) -> tuple[Image.Image, Image.Image, float | None, float | None]:
    """Crop both images so subjects appear at the same relative zoom level.
    Returns (left, right, left_face_top_frac, right_face_top_frac)."""
    lf = detect_face(left)
    rf = detect_face(right)

    if lf is not None and rf is not None:
        lr = lf[3] / left.height   # face_h / image_h
        rr = rf[3] / right.height
        target_ratio = max(lr, rr)  # zoom both in to match the closer shot
        # Use aspect ratio of the more-zoomed-in image (needs less cropping)
        aspect = (left.width / left.height) if lr >= rr else (right.width / right.height)
        print(f"    faces detected — target zoom: {target_ratio:.0%} of frame height")
        left,  l_ft = _smart_crop(left,  lf, target_ratio, aspect)
        right, r_ft = _smart_crop(right, rf, target_ratio, aspect)
    else:
        missing = []
        if lf is None:
            missing.append('left')
        if rf is None:
            missing.append('right')
        print(f"    no face in {' + '.join(missing)} — center-crop fallback")
        left,  l_ft = _smart_crop(left,  None, None, FALLBACK_AR)
        right, r_ft = _smart_crop(right, None, None, FALLBACK_AR)

    return left, right, l_ft, r_ft


# ── Label overlay ─────────────────────────────────────────────────────────────

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in [
        '/System/Library/Fonts/Helvetica.ttc',
        '/System/Library/Fonts/HelveticaNeue.ttc',
        '/Library/Fonts/Arial.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def add_label(img: Image.Image, date_str: str, location: str) -> Image.Image:
    font_d = _load_font(FONT_DATE_SZ)
    font_l = _load_font(FONT_LOC_SZ)

    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    _, _, _, h_d = draw.textbbox((0, 0), date_str or ' ', font=font_d)
    _, _, _, h_l = draw.textbbox((0, 0), location or ' ', font=font_l)
    strip_h = h_d + h_l + LABEL_PAD * 3
    y0 = img.height - strip_h

    draw.rectangle([(0, y0), (img.width, img.height)], fill=(0, 0, 0, LABEL_ALPHA))
    if date_str:
        draw.text((LABEL_PAD, y0 + LABEL_PAD), date_str,
                  font=font_d, fill=(255, 255, 255, 255))
    if location:
        draw.text((LABEL_PAD, y0 + LABEL_PAD + h_d + LABEL_PAD // 2), location,
                  font=font_l, fill=(210, 210, 210, 255))

    return Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')


# ── Composite ─────────────────────────────────────────────────────────────────

def make_composite(left: Image.Image, right: Image.Image,
                   l_face_top: float | None = None,
                   r_face_top: float | None = None) -> Image.Image:
    def to_height(im: Image.Image, h: int) -> Image.Image:
        return im.resize((int(im.width * h / im.height), h), Image.LANCZOS)

    l = to_height(left,  TARGET_HEIGHT)
    r = to_height(right, TARGET_HEIGHT)

    l_off = r_off = 0
    if l_face_top is not None and r_face_top is not None:
        lp = int(l_face_top * TARGET_HEIGHT)
        rp = int(r_face_top * TARGET_HEIGHT)
        if lp < rp:
            r_off = rp - lp
        elif rp < lp:
            l_off = lp - rp
        if l_off or r_off:
            print(f"    head alignment: offset {max(l_off, r_off)}px "
                  f"({'left' if l_off else 'right'} panel padded)")

    canvas_h = TARGET_HEIGHT + max(l_off, r_off)
    canvas = Image.new('RGB', (l.width + GAP_PX + r.width, canvas_h), (20, 20, 20))
    canvas.paste(l, (0, l_off))
    canvas.paste(r, (l.width + GAP_PX, r_off))
    return canvas


# ── Processing ────────────────────────────────────────────────────────────────

def process_pair(prefix: str, files: list[Path], output_dir: Path) -> None:
    print(f"\n{prefix}")

    items = [(f, get_exif(f)) for f in files]
    items.sort(key=lambda x: x[1].get('DateTimeOriginal', ''))
    (lp, le), (rp, re_) = items
    print(f"  left:  {lp.name}")
    print(f"  right: {rp.name}")

    left  = open_image(lp)
    right = open_image(rp)

    print("  matching zoom...")
    left, right, l_face_top, r_face_top = match_zoom(left, right)

    l_date, l_year = format_date(le)
    r_date, r_year = format_date(re_)
    l_loc = r_loc = ''

    if gps := to_decimal_gps(le):
        print(f"  geocoding left...")
        l_loc = reverse_geocode(*gps)
        print(f"    {l_loc}")
    if gps := to_decimal_gps(re_):
        print(f"  geocoding right...")
        r_loc = reverse_geocode(*gps)
        print(f"    {r_loc}")

    left  = add_label(left,  l_year, l_loc)
    right = add_label(right, r_year, r_loc)

    composite = make_composite(left, right, l_face_top, r_face_top)

    # Filename: "{location} {year1}-{year2}.jpg"
    # Use left location; fall back to right; fall back to prefix
    place = l_loc or r_loc or prefix
    years = f"{l_year}-{r_year}" if l_year and r_year else (l_year or r_year or '')
    filename = f"{place} {years}.jpg".strip()
    out = output_dir / filename
    composite.save(out, 'JPEG', quality=92)
    print(f"  saved: {out.name}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Create side-by-side composites from recreation photo pairs.'
    )
    parser.add_argument('folder', nargs='?', default='.',
                        help='Folder containing photo pairs (default: current directory)')
    parser.add_argument('-o', '--output',
                        help='Output folder (default: composites/ inside FOLDER)')
    args = parser.parse_args()

    folder     = Path(args.folder).expanduser().resolve()
    output_dir = (Path(args.output).expanduser().resolve()
                  if args.output else folder / 'composites')
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(folder)
    orphans = [
        f for f in sorted(folder.iterdir())
        if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
        and not f.name.startswith('.')
        and extract_prefix(f.stem) not in pairs
    ]

    print(f"Folder: {folder}")
    print(f"{len(pairs)} pair(s) found"
          + (f", {len(orphans)} unpaired file(s) skipped" if orphans else ''))
    for f in orphans:
        print(f"  skip: {f.name}")

    for prefix, files in sorted(pairs.items()):
        process_pair(prefix, files, output_dir)

    print(f"\nDone. Output -> {output_dir}")


if __name__ == '__main__':
    main()
