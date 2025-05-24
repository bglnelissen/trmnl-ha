#!/usr/bin/env python3

"""
Test script to generate screenshots with different combinations of:
- Contrast (0.1 to 2.0)
- Brightness (0.1 to 2.0)
- Dithering algorithms
"""

import subprocess
import numpy as np
from pathlib import Path

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

# Ensure tests directory exists
Path('tests').mkdir(exist_ok=True)

# Total number of combinations
total = len(contrasts) * len(brightnesses) * len(dither_algorithms)
current = 0

print(f"Will generate {total} test images...")
print(f"Contrasts: {contrasts}")
print(f"Brightnesses: {brightnesses}")
print(f"Dither algorithms: {dither_algorithms}")
print()

for contrast in contrasts:
    for brightness in brightnesses:
        for dither in dither_algorithms:
            current += 1
            output_file = f"tests/c{contrast}_b{brightness}_{dither}.png"
            
            # Skip if file already exists
            if Path(output_file).exists():
                print(f"[{current}/{total}] Skipping existing {output_file}")
                continue
            
            print(f"[{current}/{total}] Generating {output_file}")
            
            cmd = [
                'python3', 'ha_screenshot.py',
                '--output', output_file,
                '--contrast', str(contrast),
                '--brightness', str(brightness),
                '--dither', dither
            ]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error generating {output_file}: {e}")
                continue

print("\nTest generation complete!")
print(f"Images are saved in the 'tests' directory")
print("Filename format: c[contrast]_b[brightness]_[dither].png") 