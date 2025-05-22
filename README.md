# Home Assistant Screenshot Tool

A Python script that automatically logs into a Home Assistant instance and takes a screenshot of the interface.

## Requirements

- Python 3.7+
- Playwright
- Home Assistant instance

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd trmnl-ha
```

2. Install dependencies:
```bash
pip install playwright
playwright install chromium
```

## Usage

Run the script with the following command:

```bash
python3 ha_screenshot.py <home-assistant-url> -u <username> -p <password>
```

Example:
```bash
python3 ha_screenshot.py http://homeassistant.local:8123 -u admin -p password
```

Options:
- `-o, --output`: Output file path (default: ha_screenshot.png)
- `-u, --username`: Home Assistant username (required)
- `-p, --password`: Home Assistant password (required)

## Features

- Automated login to Home Assistant
- Configurable viewport size (800x480)
- SSL/HTTPS support
- Custom user agent
- Error handling and debugging output 