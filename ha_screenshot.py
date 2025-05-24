#!/usr/bin/env python3

"""
Home Assistant Screenshot Tool

This script automates the process of taking screenshots of a Home Assistant instance.
It handles authentication, navigation, and screenshot capture using Playwright.
The output is optimized for e-ink displays with black and white colors.

Author: Bastiaan Nelissen
License: MIT
Copyright (c) 2024 Bastiaan Nelissen
Repository: https://github.com/bglnelissen/trmnl-ha
"""

import asyncio
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import yaml
from PIL import Image, ImageEnhance, ImageOps
import tempfile
import time
import numpy as np

from playwright.async_api import async_playwright, Browser, Page, Response, TimeoutError

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Available dithering algorithms
DITHER_ALGORITHMS = {
    'floyd-steinberg': {
        'method': Image.Dither.FLOYDSTEINBERG,
        'description': 'Floyd-Steinberg dithering - Good all-around dithering with error diffusion'
    },
    'ordered': {
        'method': Image.Dither.ORDERED,
        'description': 'Ordered dithering - Creates a regular pattern'
    },
    'threshold': {
        'method': Image.Dither.NONE,
        'description': 'Simple threshold - No dithering (pure black and white)'
    },
    'none': {
        'method': Image.Dither.NONE,
        'description': 'Alias for threshold - No dithering (pure black and white)'
    },
    'halftone': {
        'method': None,  # Custom implementation
        'description': 'Halftone dithering - Simulates newspaper-style dots'
    },
    'bayer': {
        'method': None,  # Custom implementation
        'description': 'Bayer dithering - Ordered dithering with Bayer matrix'
    },
    'error-diffusion': {
        'method': Image.Dither.RASTERIZE,
        'description': 'Error diffusion dithering - Alternative to Floyd-Steinberg'
    }
}

class ImageProcessor:
    """Handles image processing operations."""
    
    @staticmethod
    def create_bayer_matrix(size: int) -> np.ndarray:
        """Create a Bayer dithering matrix of given size."""
        # Start with 2x2 Bayer matrix
        bayer = np.array([[0, 2],
                         [3, 1]])
        
        # Expand to desired size
        n = 2
        while n < size:
            bayer = np.repeat(np.repeat(bayer, 2, axis=0), 2, axis=1)
            bayer = bayer * 4
            n *= 2
            offset = np.array([[0, 2],
                             [3, 1]])
            for i in range(0, n, 2):
                for j in range(0, n, 2):
                    bayer[i:i+2, j:j+2] += offset
        
        # Normalize to 0-255 range
        return (bayer * 255 / (size * size)).astype(np.uint8)

    @staticmethod
    def apply_bayer_dithering(img: Image.Image, matrix_size: int = 8) -> Image.Image:
        """Apply Bayer dithering to an image."""
        # Convert image to numpy array
        img_array = np.array(img)
        
        # Create and tile Bayer matrix
        bayer = ImageProcessor.create_bayer_matrix(matrix_size)
        h, w = img_array.shape
        bayer_tiled = np.tile(bayer, (h // matrix_size + 1, w // matrix_size + 1))
        bayer_tiled = bayer_tiled[:h, :w]
        
        # Apply dithering
        result = np.where(img_array > bayer_tiled, 255, 0)
        
        return Image.fromarray(result.astype(np.uint8))

    @staticmethod
    def apply_halftone_dithering(img: Image.Image) -> Image.Image:
        """Apply halftone dithering to an image."""
        # Convert image to numpy array
        img_array = np.array(img)
        h, w = img_array.shape
        
        # Create output array
        output = np.zeros((h, w), dtype=np.uint8)
        
        # Define halftone pattern (2x2)
        patterns = [
            np.array([[0, 0], [0, 0]]),  # 0%
            np.array([[255, 0], [0, 0]]),  # 25%
            np.array([[255, 0], [0, 255]]),  # 50%
            np.array([[255, 255], [0, 255]]),  # 75%
            np.array([[255, 255], [255, 255]])  # 100%
        ]
        
        # Process image in 2x2 blocks
        for i in range(0, h-1, 2):
            for j in range(0, w-1, 2):
                block = img_array[i:i+2, j:j+2]
                avg = np.mean(block)
                
                # Select appropriate pattern
                pattern_idx = int(avg * len(patterns) / 256)
                pattern_idx = min(pattern_idx, len(patterns) - 1)
                
                output[i:i+2, j:j+2] = patterns[pattern_idx]
        
        return Image.fromarray(output)

    @staticmethod
    def adjust_contrast_brightness(img: Image.Image, contrast: float, brightness: float) -> Image.Image:
        """
        Adjust image contrast and brightness.
        
        Args:
            img: Input image
            contrast: Contrast factor (1.0 = unchanged)
            brightness: Brightness factor (1.0 = unchanged)
            
        Returns:
            Adjusted image
        """
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)
        
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
        
        return img

    @staticmethod
    def apply_dithering(img: Image.Image, algorithm: str, threshold_value: int = 128) -> Image.Image:
        """
        Apply specified dithering algorithm to image.
        
        Args:
            img: Input image
            algorithm: Name of dithering algorithm to use
            threshold_value: Threshold value for black/white conversion (0-255, default 128)
            
        Returns:
            Dithered image
        """
        if algorithm not in DITHER_ALGORITHMS:
            raise ValueError(f"Unknown dithering algorithm: {algorithm}")
        
        algo_info = DITHER_ALGORITHMS[algorithm]
        
        if algorithm == 'halftone':
            return ImageProcessor.apply_halftone_dithering(img)
        elif algorithm == 'bayer':
            return ImageProcessor.apply_bayer_dithering(img)
        elif algorithm in ['threshold', 'none']:
            # Convert to numpy array for custom thresholding
            img_array = np.array(img)
            result = np.where(img_array > threshold_value, 255, 0)
            return Image.fromarray(result.astype(np.uint8))
        else:
            return img.convert('1', dither=algo_info['method'])

class HomeAssistantScreenshotter:
    """Handles the automation of taking screenshots from a Home Assistant instance."""
    
    def __init__(self, settings: Dict[str, Any], output: str):
        """
        Initialize the screenshotter with settings.
        
        Args:
            settings: Dictionary containing all settings (url, credentials, display settings)
            output: Path where the screenshot should be saved
        """
        self.url = settings['url'].rstrip('/')
        self.output = output
        self.username = settings['username']
        self.password = settings['password']
        self.zoom = float(settings.get('zoom', 1.0))
        self.width = int(settings.get('width', 800))
        self.height = int(settings.get('height', 480))
        self.scroll_offset = int(settings.get('scroll_offset', 0))
        self.timeout = int(settings.get('timeout', 30)) * 1000  # Convert to milliseconds
        self.retry_count = int(settings.get('retry_count', 3))
        self.retry_delay = int(settings.get('retry_delay', 5))
        self.wait_time = int(settings.get('wait_time', 10))
        self.wait_for_no_activity = int(settings.get('wait_for_no_activity', 2))
        self.wait_for_selector = str(settings.get('wait_for_selector', 'ha-panel-lovelace'))
        self.contrast = float(settings.get('contrast', 1.0))
        self.brightness = float(settings.get('brightness', 1.0))
        self.dither_algorithm = str(settings.get('dither', 'floyd-steinberg')).lower()
        self.threshold_value = int(settings.get('threshold_value', 128))  # Default middle point (0-255)
        self.invert = bool(settings.get('invert', False))
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self._last_network_activity = 0
        self._network_idle = False

    async def __aenter__(self):
        """Set up the browser context when entering the async context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting the async context."""
        if self.browser:
            await self.browser.close()

    async def setup_browser(self) -> None:
        """Initialize the browser with appropriate settings."""
        playwright = await async_playwright().start()
        
        # Calculate viewport size based on zoom level and add extra height for scrolling
        viewport_width = int(self.width / self.zoom)
        viewport_height = int((self.height + self.scroll_offset) / self.zoom)
        
        # Launch browser with specific configurations
        self.browser = await playwright.chromium.launch(
            args=['--disable-gpu', '--no-sandbox'],
            headless=True
        )
        
        # Create a browser context with custom viewport and settings
        context = await self.browser.new_context(
            viewport={'width': viewport_width, 'height': viewport_height},
            ignore_https_errors=True,
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36'
        )
        
        self.page = await context.new_page()
        
        # Set default timeout
        self.page.set_default_timeout(self.timeout)
        
        # Set zoom level
        if self.zoom != 1.0:
            logger.info(f"Setting zoom level to {self.zoom * 100}%")
            await self.page.set_viewport_size({
                'width': viewport_width,
                'height': viewport_height
            })
            await self.page.evaluate(f'document.body.style.zoom = {self.zoom}')

    async def handle_auth_response(self, response: Response) -> None:
        """
        Handle authentication response for debugging purposes.
        
        Args:
            response: The response from an authentication request
        """
        if "/auth/login_flow/" in response.url:
            try:
                response_data = await response.json()
                logger.debug(f"Auth response: {response_data}")
            except Exception as e:
                logger.debug(f"Could not parse auth response: {e}")

    async def login(self) -> bool:
        """
        Handle the login process if credentials are provided.
        
        Returns:
            bool: True if login was successful, False otherwise
        """
        if not (self.username and self.password):
            logger.info("No credentials provided, skipping login")
            return False

        for attempt in range(self.retry_count):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt + 1} of {self.retry_count}")
                    # Wait before retrying
                    await asyncio.sleep(self.retry_delay)
                    # Reload the page for a fresh attempt
                    await self.page.reload()

                # Wait for the login form
                logger.info("Waiting for login form...")
                await self.page.wait_for_selector('ha-auth-flow', timeout=self.timeout)

                # Prepare login payload
                login_payload = {
                    "username": self.username,
                    "password": self.password,
                    "client_id": self.url
                }

                # Set up response handler for debugging
                self.page.on("response", self.handle_auth_response)

                # Fill in credentials
                logger.info("Filling in credentials...")
                await self.page.fill('input[name="username"]', self.username)
                await self.page.fill('input[name="password"]', self.password)

                # Submit login form
                logger.info("Submitting login form...")
                await self.page.click('mwc-button')

                # Wait for successful login
                logger.info("Waiting for Home Assistant interface...")
                await self.page.wait_for_selector('home-assistant', timeout=self.timeout)
                logger.info("Successfully logged in!")
                return True

            except TimeoutError as e:
                logger.warning(f"Login attempt {attempt + 1} timed out: {e}")
                if attempt + 1 == self.retry_count:
                    logger.error("All login attempts failed")
                    return False
            except Exception as e:
                logger.error(f"Login attempt {attempt + 1} failed: {e}")
                if attempt + 1 == self.retry_count:
                    return False

        return False

    def convert_to_bw(self, temp_path: str) -> None:
        """
        Convert the screenshot to black and white for e-ink display.
        
        Args:
            temp_path: Path to the temporary color screenshot
            
        This method will convert the image to black and white using
        the specified dithering algorithm and image adjustments.
        """
        try:
            # Open the image
            with Image.open(temp_path) as img:
                # Crop the image to remove the top offset and get the desired height
                if self.scroll_offset > 0:
                    img = img.crop((0, self.scroll_offset, img.width, self.scroll_offset + self.height))
                
                # Convert to grayscale first
                img_gray = img.convert('L')
                
                # Apply contrast and brightness adjustments
                img_adjusted = ImageProcessor.adjust_contrast_brightness(
                    img_gray, self.contrast, self.brightness
                )
                
                # Apply dithering
                img_dithered = ImageProcessor.apply_dithering(
                    img_adjusted, 
                    self.dither_algorithm,
                    self.threshold_value
                )
                
                # Invert if requested
                if self.invert:
                    img_dithered = ImageOps.invert(img_dithered)
                
                # Ensure the final size matches the output dimensions
                if img_dithered.size != (self.width, self.height):
                    img_dithered = img_dithered.resize((self.width, self.height), Image.LANCZOS)
                
                # Save the black and white image
                img_dithered.save(self.output, 'PNG', optimize=True)
                
            logger.info("Successfully converted to black and white")
            
        except Exception as e:
            logger.error(f"Failed to convert image to black and white: {e}")
            raise

    async def take_screenshot(self) -> None:
        """
        Take a screenshot of the Home Assistant interface and convert to black and white.
        
        Raises:
            Exception: If screenshot capture fails
        """
        try:
            # Create a temporary file for the initial screenshot
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                
            # Take the initial screenshot in color
            logger.info("Taking initial screenshot...")
            await self.page.screenshot(path=temp_path)
            
            # Convert to black and white
            logger.info("Converting to black and white...")
            self.convert_to_bw(temp_path)
            
            # Clean up the temporary file
            Path(temp_path).unlink()
            
            logger.info(f"Screenshot saved to {self.output}")
            
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            raise

    async def _handle_network_activity(self, request):
        """Track network activity to determine when page is fully loaded."""
        self._last_network_activity = time.time()
        self._network_idle = False

    async def _handle_network_idle(self):
        """Handle network idle state."""
        self._network_idle = True

    async def wait_for_page_load(self):
        """Wait for the page to be fully loaded."""
        try:
            # Wait for the specified selector
            logger.info(f"Waiting for {self.wait_for_selector} element...")
            await self.page.wait_for_selector(self.wait_for_selector, timeout=self.timeout)
            
            # Wait for initial load time
            logger.info(f"Waiting {self.wait_time} seconds for page to load...")
            await asyncio.sleep(self.wait_time)
            
            # Wait for network to be idle
            if self.wait_for_no_activity > 0:
                self.page.on('request', self._handle_network_activity)
                self.page.on('requestfinished', self._handle_network_idle)
                
                start_time = time.time()
                while time.time() - start_time < self.wait_for_no_activity:
                    if self._network_idle and time.time() - self._last_network_activity >= self.wait_for_no_activity:
                        break
                    await asyncio.sleep(0.1)
                
                logger.info(f"Waited {self.wait_for_no_activity} seconds after last network activity")
            
        except Exception as e:
            logger.warning(f"Wait condition not met: {e}")
            logger.info("Proceeding with screenshot anyway")

    async def process(self) -> None:
        """Main process to navigate to Home Assistant and take a screenshot."""
        try:
            # Log image processing settings
            logger.info("Image processing settings:")
            logger.info(f"  - Contrast: {self.contrast:.2f}")
            logger.info(f"  - Brightness: {self.brightness:.2f}")
            logger.info(f"  - Dithering: {self.dither_algorithm} ({DITHER_ALGORITHMS[self.dither_algorithm]['description']})")
            if self.dither_algorithm in ['threshold', 'none']:
                logger.info(f"  - Threshold value: {self.threshold_value}")
            logger.info(f"  - Invert colors: {self.invert}")
            
            # Set up the browser
            await self.setup_browser()
            
            # Navigate to Home Assistant
            logger.info(f"Navigating to {self.url}")
            await self.page.goto(self.url, timeout=self.timeout)

            # Handle login if credentials provided
            if self.username and self.password:
                login_success = await self.login()
                if not login_success:
                    logger.warning("Proceeding with screenshot despite login issues")
            
            # Wait for page to load completely
            await self.wait_for_page_load()

            # Take the screenshot
            await self.take_screenshot()

        except Exception as e:
            logger.error(f"Process failed: {e}")
            raise
        finally:
            if self.browser:
                await self.browser.close()

def load_settings(settings_file: str = 'settings.yaml') -> Dict[str, Any]:
    """
    Load settings from the YAML file.
    
    Args:
        settings_file: Path to the settings file
        
    Returns:
        Dict containing all settings
        
    Raises:
        FileNotFoundError: If settings file doesn't exist
        yaml.YAMLError: If settings file is invalid
    """
    try:
        with open(settings_file, 'r') as f:
            settings = yaml.safe_load(f)
            
        # Required settings
        required_fields = ['url', 'username', 'password']
        missing_fields = [field for field in required_fields if field not in settings]
        
        if missing_fields:
            raise ValueError(f"Missing required fields in settings file: {', '.join(missing_fields)}")
        
        # Set default values for optional settings
        settings.setdefault('zoom', 1.0)
        settings.setdefault('width', 800)
        settings.setdefault('height', 480)
        settings.setdefault('scroll_offset', 0)
        settings.setdefault('timeout', 30)
        settings.setdefault('retry_count', 3)
        settings.setdefault('retry_delay', 5)
        settings.setdefault('wait_time', 10)
        settings.setdefault('wait_for_no_activity', 2)
        settings.setdefault('wait_for_selector', 'ha-panel-lovelace')
        settings.setdefault('contrast', 1.0)
        settings.setdefault('brightness', 1.0)
        settings.setdefault('dither', 'floyd-steinberg')
        settings.setdefault('threshold_value', 128)
        settings.setdefault('invert', False)
            
        return settings
    except FileNotFoundError:
        logger.error(f"Settings file '{settings_file}' not found")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing settings file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
        raise

def parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Take a screenshot of Home Assistant',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-o', '--output', default='ha_screenshot.png',
                      help='Output file path')
    parser.add_argument('-s', '--settings', default='settings.yaml',
                      help='Path to settings file')
    parser.add_argument('--scroll_offset', type=int,
                      help='Number of pixels to scroll down from top (overrides settings.yaml)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    parser.add_argument('--contrast', type=float,
                      help='Contrast adjustment (1.0 = normal, overrides settings.yaml)')
    parser.add_argument('--brightness', type=float,
                      help='Brightness adjustment (1.0 = normal, overrides settings.yaml)')
    parser.add_argument('--dither', choices=list(DITHER_ALGORITHMS.keys()),
                      help='Dithering algorithm to use (overrides settings.yaml)')
    parser.add_argument('--threshold-value', type=int, choices=range(0, 256),
                      help='Threshold value for black/white conversion (0-255, default 128, only used with threshold/none dithering)')
    parser.add_argument('--invert', action='store_true',
                      help='Invert colors (overrides settings.yaml)')
    parser.add_argument('--list-dither', action='store_true',
                      help='List available dithering algorithms and exit')
    
    args = parser.parse_args()
    
    if args.list_dither:
        print("\nAvailable dithering algorithms:")
        for name, info in DITHER_ALGORITHMS.items():
            print(f"\n{name}:")
            print(f"  {info['description']}")
        exit(0)
    
    return args

async def main() -> None:
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Set logging level based on debug flag
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load settings
    try:
        settings = load_settings(args.settings)
        
        # Override settings with command line arguments
        if args.scroll_offset is not None:
            settings['scroll_offset'] = args.scroll_offset
        if args.contrast is not None:
            settings['contrast'] = args.contrast
        if args.brightness is not None:
            settings['brightness'] = args.brightness
        if args.dither:
            settings['dither'] = args.dither
        if args.threshold_value is not None:
            settings['threshold_value'] = args.threshold_value
        if args.invert:
            settings['invert'] = True
            
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        return
    
    async with HomeAssistantScreenshotter(
        settings=settings,
        output=args.output
    ) as screenshotter:
        await screenshotter.process()

if __name__ == "__main__":
    asyncio.run(main()) 