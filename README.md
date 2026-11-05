# recreations

Creates side-by-side composite images from recreation photo pairs.

## How it works

Pairs photos by matching the filename prefix before the trailing date:

```
Kiel - 20060116.jpg  +  Kiel - 20190428.jpg   ->  Kiel - composite.jpg
1 - 2004-12-23 ...   +  1 - 2021-12-25 ...    ->  1 - composite.jpg
```

- Older photo on the left, recreation on the right
- Face detection matches zoom levels between the two shots
- Date and location (from GPS + OpenStreetMap) overlaid at the bottom of each photo
- Outputs to `composites/` subfolder by default

## Setup

```bash
pip install -r requirements.txt
brew install exiftool   # if not already installed
```

## Usage

```bash
# Process current folder
python recreations.py

# Process a specific folder
python recreations.py "/path/to/01 - Recreations"

# Save composites to a different folder
python recreations.py "/path/to/01 - Recreations" -o ~/Desktop/output
```

## Notes

- Supported formats: JPG, HEIC, JPEG, PNG, TIFF
- HEIC files require `pillow-heif` (installed via requirements.txt); falls back to macOS `sips` if unavailable
- Geocoding uses OpenStreetMap Nominatim (free, no API key needed, 1 req/s rate limit)
- Unpaired files (no matching prefix partner) are skipped and reported
