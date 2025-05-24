# Home Assistant Screenshot Tool

A Python tool to take automated screenshots of Home Assistant dashboards, optimized for e-ink displays.

## Features

- Automated screenshot capture of Home Assistant dashboards
- Black and white conversion with multiple dithering algorithms
- Contrast and brightness adjustment
- Configurable wait times for proper page loading
- Multiple dithering options:
  - Floyd-Steinberg (default) - Good all-around dithering
  - Ordered dithering - Regular pattern
  - Threshold/None - Pure black and white, no dithering
  - Halftone - Newspaper-style dots
  - Bayer matrix - Ordered dithering pattern
  - Error diffusion - Alternative to Floyd-Steinberg
- Color inversion option
- Configurable image dimensions and cropping
- Secure credential handling

## Installation

1. Clone this repository:
```bash
git clone https://github.com/bglnelissen/trmnl-ha.git
cd trmnl-ha
```

2. Install dependencies:
```bash
pip install playwright pillow pyyaml numpy
playwright install chromium
```

## Configuration

Copy the example settings file and edit it with your configuration:

```bash
cp settings.yaml.example settings.yaml
```

Edit `settings.yaml` with your Home Assistant details and preferences:

```yaml
# Connection settings
url: http://homeassistant.local:8123
username: your_username
password: your_password

# Image processing settings
contrast: 1.2    # More contrast for e-ink
brightness: 1.0  # Normal brightness
dither: "floyd-steinberg"  # Dithering algorithm
```

## Usage

Basic usage:
```bash
python3 ha_screenshot.py
```

With options:
```bash
python3 ha_screenshot.py --output dashboard.png --contrast 1.2 --dither bayer
```

Show all options:
```bash
python3 ha_screenshot.py --help
```

List available dithering algorithms:
```bash
python3 ha_screenshot.py --list-dither
```

## Settings

All settings can be configured either in `settings.yaml` or via command line arguments:

### Connection Settings
- `url`: Your Home Assistant URL
- `username`: Your Home Assistant username
- `password`: Your Home Assistant password
- `timeout`: Connection timeout in seconds
- `retry_count`: Number of login retry attempts
- `retry_delay`: Delay between retries in seconds

### Page Load Settings
- `wait_time`: Time to wait for page load (seconds)
- `wait_for_no_activity`: Additional wait after last network activity
- `wait_for_selector`: DOM element to wait for

### Display Settings
- `zoom`: Zoom level (e.g., 0.8 for 80%)
- `width`: Output width in pixels
- `height`: Output height in pixels
- `scroll_offset`: Pixels to crop from top

### Image Processing
- `contrast`: Contrast adjustment (1.0 = normal)
- `brightness`: Brightness adjustment (1.0 = normal)
- `dither`: Dithering algorithm
- `invert`: Invert colors (true/false)

## License

MIT License - Copyright (c) 2024 Bastiaan Nelissen 