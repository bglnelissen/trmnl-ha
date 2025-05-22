# Home Assistant Screenshot Tool

A Python script that automatically logs into a Home Assistant instance and takes a screenshot of the interface. The script is designed to create black and white screenshots optimized for e-ink displays.

## Features

- Automatic login to Home Assistant
- Screenshot capture at specified dimensions
- Black and white conversion with dithering for e-ink displays
- Configurable zoom level and scroll offset
- Output size optimization

## Requirements

- Python 3.7+
- Playwright
- PyYAML
- Pillow
- Home Assistant instance

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd trmnl-ha
```

2. Install dependencies:
```bash
pip install -r requirements.txt
playwright install chromium
```

## Usage

### Basic Usage

Run the script with default settings:
```bash
python3 ha_screenshot.py
```

### Command Line Options

- `-o, --output`: Output file path (default: ha_screenshot.png)
- `-s, --settings`: Path to settings file (default: settings.yaml)
- `--scroll_offset`: Number of pixels to scroll down from top

Example:
```bash
python3 ha_screenshot.py -o custom.png --scroll_offset 45
```

### Settings File

Create a `settings.yaml` file with your configuration:

```yaml
# Connection settings
url: http://your-ha-instance:8123
username: your_username
password: your_password

# Output settings
zoom: 1.0  # 100% zoom level
width: 800  # Output width in pixels
height: 480 # Output height in pixels
scroll_offset: 30  # Number of pixels to scroll down from top
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Bas Nelissen

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 