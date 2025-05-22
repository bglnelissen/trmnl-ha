#!/usr/bin/env python3

import asyncio
from playwright.async_api import async_playwright, TimeoutError, Response
import argparse
from pathlib import Path
import json
import re

async def take_screenshot(url: str, output: str, username: str = None, password: str = None):
    async with async_playwright() as p:
        # Launch browser with additional options
        print("Launching browser...")
        browser = await p.chromium.launch(
            args=['--disable-gpu', '--no-sandbox'],
            headless=True
        )
        
        # Create context with specific viewport size
        context = await browser.new_context(
            viewport={'width': 800, 'height': 480},
            ignore_https_errors=True,
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36'
        )
        
        # Create a new page
        page = await context.new_page()
        
        print(f"Navigating to {url}...")
        await page.goto(url)
        
        # Wait for the login form to be ready
        print("Waiting for login form...")
        await page.wait_for_selector('ha-auth-flow')
        
        # Get the client_id from the URL
        client_id = url.rstrip('/')
        
        # Prepare the login payload
        login_payload = {
            "username": username,
            "password": password,
            "client_id": client_id
        }
        
        print("Attempting login...")
        
        # Listen for response to check login status
        async def handle_response(response):
            if "/auth/login_flow/" in response.url:
                try:
                    response_data = await response.json()
                    print(f"Auth response: {response_data}")
                except:
                    pass

        page.on("response", handle_response)
        
        # Fill in username and password
        await page.fill('input[name="username"]', username)
        await page.fill('input[name="password"]', password)
        
        # Click the login button and wait for navigation
        print("Clicking login button...")
        await page.click('mwc-button')
        
        # Wait for the main interface to load
        print("Waiting for Home Assistant interface...")
        try:
            await page.wait_for_selector('home-assistant', timeout=10000)
            print("Successfully logged in!")
        except TimeoutError:
            print("Failed to detect successful login. Taking screenshot anyway...")
        
        # Take the screenshot
        print(f"Taking screenshot and saving to {output}...")
        await page.screenshot(path=output)
        
        await browser.close()

def main():
    parser = argparse.ArgumentParser(description='Take a screenshot of Home Assistant')
    parser.add_argument('url', help='URL of Home Assistant instance')
    parser.add_argument('-o', '--output', default='ha_screenshot.png', help='Output file path')
    parser.add_argument('-u', '--username', required=True, help='Home Assistant username')
    parser.add_argument('-p', '--password', required=True, help='Home Assistant password')
    
    args = parser.parse_args()
    
    asyncio.run(take_screenshot(args.url, args.output, args.username, args.password))

if __name__ == "__main__":
    main() 