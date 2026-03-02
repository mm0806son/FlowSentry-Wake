#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration Data Preprocessing Script: Horizontally stitches frame pairs into a single image

Used for calibration data preparation in the Axelera SDK.
Concatenate consecutive frame pairs from datasets such as FlyingThings3D into [H, 2W, 3] RGB images.
These images are natively supported by the SDK's ImageReader and can be subsequently split within override_preprocess.
Usage:
    python prepare_calib_data.py --input data/calib_edgeflownet --output data/calib_merged

Input Directory Structure:
    input/
    ├── 0000/left/
    │   ├── 0000.png
    │   ├── 0001.png
    │   └── ...
    └── 0001/left/
        └── ...

Output Directory Structure:
    output/
    ├── merged_0000_0000.png  # 576 x 2048 x 3
    ├── merged_0000_0001.png
    └── ...
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse


TARGET_HEIGHT = 576
TARGET_WIDTH = 1024


def find_frame_pairs(input_dir):
    input_path = Path(input_dir)
    pairs = []
    
    for subdir in sorted(input_path.iterdir()):
        if not subdir.is_dir():
            continue
        
        left_dir = subdir / 'left'
        if not left_dir.exists():
            left_dir = subdir 
        
        images = sorted([f for f in left_dir.iterdir() 
                        if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        
        for i in range(len(images) - 1):
            pairs.append((images[i], images[i + 1]))
    
    if not pairs:
        images = sorted([f for f in input_path.iterdir() 
                        if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        for i in range(len(images) - 1):
            pairs.append((images[i], images[i + 1]))
    
    return pairs


def merge_frame_pair(frame1_path, frame2_path, target_h=TARGET_HEIGHT, target_w=TARGET_WIDTH):
    img1 = cv2.imread(str(frame1_path))
    img2 = cv2.imread(str(frame2_path))
    
    if img1 is None or img2 is None:
        print(f"警告: 无法读取 {frame1_path} 或 {frame2_path}")
        return None
    
    img1 = cv2.resize(img1, (target_w, target_h))
    img2 = cv2.resize(img2, (target_w, target_h))
    
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    merged = np.hstack((img1, img2))
    
    return merged


def main():
    parser = argparse.ArgumentParser(description='Prepare EdgeFlowNet calibration data')
    parser.add_argument('--input', '-i', default='data/calib_edgeflownet',
                       help='Input directory (containing frame sequences)')
    parser.add_argument('--output', '-o', default='data/calib_merged',
                       help='Output directory (for merged images)')
    parser.add_argument('--max-pairs', '-n', type=int, default=200,
                       help='Maximum number of frame pairs (default: 200)')
    parser.add_argument('--height', type=int, default=TARGET_HEIGHT,
                       help=f'Target height (default:{TARGET_HEIGHT})')
    parser.add_argument('--width', type=int, default=TARGET_WIDTH,
                       help=f'Target width (default: {TARGET_WIDTH})')
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Target resolution: {args.width}x{args.height}")
    print(f"Max frame pairs: {args.max_pairs}")
    print("-" * 50)
    
    pairs = find_frame_pairs(args.input)
    print(f"Found {len(pairs)} frame pairs")
    
    if not pairs:
        print("Error: No frame pairs found!")
        return
    
    pairs = pairs[:args.max_pairs]
    
    count = 0
    for i, (frame1, frame2) in enumerate(pairs):
        merged = merge_frame_pair(frame1, frame2, args.height, args.width)
        if merged is None:
            continue
        
        output_file = output_path / f"merged_{i:04d}.png"
        merged_bgr = cv2.cvtColor(merged, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_file), merged_bgr)
        count += 1
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(pairs)} frame pairs...")
    
    print("-" * 50)
    print(f"Done! Generated{count} merged images.")
    print(f"Output directory: {output_path.absolute()}")
    print(f"Image dimensions:{args.width * 2}x{args.height} (W x H)")


if __name__ == '__main__':
    main()
