#!/usr/bin/env python3

"""
Test script to generate screenshots with different combinations of:
- Contrast (0.1 to 2.0)
- Brightness (0.1 to 2.0)
- Dithering algorithms

This version uses multiprocessing to run tests in parallel.
"""

import subprocess
import numpy as np
from pathlib import Path
import multiprocessing as mp
from itertools import product
import time
import os

# Test parameters
contrasts = np.arange(0.1, 2.1, 0.3).round(1)  # 0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9
brightnesses = np.arange(0.1, 2.1, 0.3).round(1)  # 0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9
dither_algorithms = [
    'floyd-steinberg',
    'ordered',
    'threshold',
    'halftone',
    'bayer',
    'error-diffusion'
]

def process_combination(args):
    """Process a single combination of parameters."""
    contrast, brightness, dither = args
    output_file = f"tests/c{contrast}_b{brightness}_{dither}.png"
    
    # Skip if file already exists
    if Path(output_file).exists():
        print(f"Skipping existing {output_file}")
        return True
    
    print(f"Generating {output_file}")
    
    cmd = [
        'python3', 'ha_screenshot.py',
        '--output', output_file,
        '--contrast', str(contrast),
        '--brightness', str(brightness),
        '--dither', dither
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating {output_file}: {e}")
        return False

def main():
    """Main entry point for parallel test generation."""
    # Ensure tests directory exists
    Path('tests').mkdir(exist_ok=True)
    
    # Create list of all combinations
    combinations = list(product(contrasts, brightnesses, dither_algorithms))
    total = len(combinations)
    
    print(f"Will generate {total} test images using {mp.cpu_count()} processes...")
    print(f"Contrasts: {contrasts}")
    print(f"Brightnesses: {brightnesses}")
    print(f"Dither algorithms: {dither_algorithms}")
    print()
    
    # Start timing
    start_time = time.time()
    
    # Create a pool of workers and process combinations in parallel
    with mp.Pool() as pool:
        results = list(pool.imap_unordered(process_combination, combinations))
    
    # Calculate statistics
    successful = sum(results)
    failed = total - successful
    duration = time.time() - start_time
    
    print("\nTest generation complete!")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Total combinations: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Images are saved in the 'tests' directory")
    print("Filename format: c[contrast]_b[brightness]_[dither].png")

if __name__ == "__main__":
    main() 