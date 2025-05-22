#!/usr/bin/env python3

"""
Home Assistant Screenshot Tool

This script automates the process of taking screenshots of a Home Assistant instance.
It handles authentication, navigation, and screenshot capture using Playwright.
The output is optimized for e-ink displays with black and white colors.
"""

import asyncio
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from PIL import Image
import tempfile

from playwright.async_api import async_playwright, Browser, Page, Response, TimeoutError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

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

        try:
            # Wait for the login form
            logger.info("Waiting for login form...")
            await self.page.wait_for_selector('ha-auth-flow', timeout=10000)

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
            await self.page.wait_for_selector('home-assistant', timeout=10000)
            logger.info("Successfully logged in!")
            return True

        except TimeoutError:
            logger.warning("Login process timed out")
            return False
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False

    def convert_to_bw(self, temp_path: str) -> None:
        """
        Convert the screenshot to black and white for e-ink display.
        
        Args:
            temp_path: Path to the temporary color screenshot
            
        This method will convert the image to black and white using
        dithering for better results on e-ink displays.
        """
        try:
            # Open the image
            with Image.open(temp_path) as img:
                # Crop the image to remove the top offset and get the desired height
                if self.scroll_offset > 0:
                    img = img.crop((0, self.scroll_offset, img.width, self.scroll_offset + self.height))
                
                # Convert to grayscale first
                img_gray = img.convert('L')
                
                # Convert to black and white using dithering
                # The dithering helps maintain detail in the conversion
                img_bw = img_gray.convert('1', dither=Image.FLOYDSTEINBERG)
                
                # Ensure the final size matches the output dimensions
                if img_bw.size != (self.width, self.height):
                    img_bw = img_bw.resize((self.width, self.height), Image.LANCZOS)
                
                # Save the black and white image
                img_bw.save(self.output, 'PNG', optimize=True)
                
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

    async def process(self) -> None:
        """Main process to navigate to Home Assistant and take a screenshot."""
        try:
            # Set up the browser
            await self.setup_browser()
            
            # Navigate to Home Assistant
            logger.info(f"Navigating to {self.url}")
            await self.page.goto(self.url)

            # Handle login if credentials provided
            if self.username and self.password:
                login_success = await self.login()
                if not login_success:
                    logger.warning("Proceeding with screenshot despite login issues")

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
    
    return parser.parse_args()

async def main() -> None:
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load settings
    try:
        settings = load_settings(args.settings)
        
        # Override scroll_offset if provided in command line
        if args.scroll_offset is not None:
            settings['scroll_offset'] = args.scroll_offset
            
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