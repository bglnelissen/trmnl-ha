# Home Assistant Screenshot Tool

A Python script that automatically logs into a Home Assistant instance and takes a screenshot of the interface. The script is designed to create black and white screenshots optimized for e-ink displays.

## Features

- Automatic login to Home Assistant
- Screenshot capture at specified dimensions
- Black and white conversion with dithering for e-ink displays
- Configurable zoom level and scroll offset
- Output size optimization
- Secure settings management

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

3. Set up your configuration:
```bash
cp settings.yaml.example settings.yaml
# Edit settings.yaml with your credentials and preferences
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
- `--scroll_offset`: Number of pixels to scroll down from top (overrides settings.yaml)

Examples:
```bash
# Custom output file
python3 ha_screenshot.py -o custom.png

# Different scroll offset
python3 ha_screenshot.py --scroll_offset 45

# Custom settings file
python3 ha_screenshot.py -s custom_settings.yaml
```

### Settings File

Copy `settings.yaml.example` to `settings.yaml` and customize it:

```yaml
# Connection settings
url: http://your-ha-instance:8123
username: your_username
password: your_password

# Output settings
zoom: 1.0       # Zoom level (e.g., 0.8 for 80%)
width: 800      # Output width in pixels
height: 480     # Output height in pixels
scroll_offset: 30 # Number of pixels to crop from top
```

### Output Customization

- **Zoom Level**: Adjust the `zoom` setting to capture more or less of the interface
- **Scroll Offset**: Remove unwanted top portions of the interface (e.g., headers)
- **Dimensions**: Customize `width` and `height` for your specific display
- **Black & White**: The output is automatically optimized for e-ink displays using dithering

## Security

- The `settings.yaml` file is ignored by git to prevent accidental credential exposure
- Use the example file (`settings.yaml.example`) as a template
- Keep your credentials secure and never commit them to version control

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Bas Nelissen

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. When contributing:

1. Fork the repository
2. Create a new branch for your feature
3. Add your changes
4. Submit a pull request

Please ensure your code follows the existing style and includes appropriate documentation. 