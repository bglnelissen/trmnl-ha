#!/usr/bin/env python3

"""
Home Assistant Screenshot Tool

This script automates the process of taking screenshots of a Home Assistant instance.
It handles authentication, navigation, and screenshot capture using Playwright.
"""

import asyncio
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

from playwright.async_api import async_playwright, Browser, Page, Response, TimeoutError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HomeAssistantScreenshotter:
    """Handles the automation of taking screenshots from a Home Assistant instance."""
    
    def __init__(self, url: str, output: str, username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize the screenshotter with connection details.
        
        Args:
            url: The URL of the Home Assistant instance
            output: Path where the screenshot should be saved
            username: Home Assistant username (optional)
            password: Home Assistant password (optional)
        """
        self.url = url.rstrip('/')
        self.output = output
        self.username = username
        self.password = password
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
        
        # Launch browser with specific configurations
        self.browser = await playwright.chromium.launch(
            args=['--disable-gpu', '--no-sandbox'],
            headless=True
        )
        
        # Create a browser context with custom viewport and settings
        context = await self.browser.new_context(
            viewport={'width': 800, 'height': 480},
            ignore_https_errors=True,
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36'
        )
        
        self.page = await context.new_page()

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

    async def take_screenshot(self) -> None:
        """
        Take a screenshot of the Home Assistant interface.
        
        Raises:
            Exception: If screenshot capture fails
        """
        try:
            logger.info(f"Taking screenshot and saving to {self.output}")
            await self.page.screenshot(path=self.output)
            logger.info("Screenshot saved successfully")
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

def load_credentials(credentials_file: str = 'credentials.yaml') -> Dict[str, str]:
    """
    Load credentials from the YAML file.
    
    Args:
        credentials_file: Path to the credentials file
        
    Returns:
        Dict containing url, username, and password
        
    Raises:
        FileNotFoundError: If credentials file doesn't exist
        yaml.YAMLError: If credentials file is invalid
    """
    try:
        with open(credentials_file, 'r') as f:
            credentials = yaml.safe_load(f)
            
        required_fields = ['url', 'username', 'password']
        missing_fields = [field for field in required_fields if field not in credentials]
        
        if missing_fields:
            raise ValueError(f"Missing required fields in credentials file: {', '.join(missing_fields)}")
            
        return credentials
    except FileNotFoundError:
        logger.error(f"Credentials file '{credentials_file}' not found")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing credentials file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading credentials: {e}")
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
    parser.add_argument('-c', '--credentials', default='credentials.yaml',
                      help='Path to credentials file')
    
    return parser.parse_args()

async def main() -> None:
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load credentials
    try:
        credentials = load_credentials(args.credentials)
    except Exception as e:
        logger.error(f"Failed to load credentials: {e}")
        return
    
    async with HomeAssistantScreenshotter(
        url=credentials['url'],
        output=args.output,
        username=credentials['username'],
        password=credentials['password']
    ) as screenshotter:
        await screenshotter.process()

if __name__ == "__main__":
    asyncio.run(main()) 