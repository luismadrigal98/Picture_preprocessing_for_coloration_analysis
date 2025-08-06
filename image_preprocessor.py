#!/usr/bin/env python3

'''
Main cli program to preprocess images for coloration analysis.

This entry point wil integrate the three main steps used in this study to analyse coloration:

- Batch normalization of names (filenames)
- Rotation of images to locate the colorchecker in the left bottom corner
- Cropping and masking of the images

@author: Luis Javier Madrigal-Roca
@date: 2025-08-06

'''

import argparse
import os
from pathlib import Path

# Configuring logging
import logging
logger = logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add src directory to the path to import modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.picture_batch_consolidation import *
from src.auto_rotation_images import *
from src.simple_flower_crop import *

def main():
    
    # Build the main parser (it will contain three different subparsers)
    parser = argparse.ArgumentParser(description="Preprocess images for coloration analysis.")

    # Global arguments
    parser.add_argument('--debug', '-d', action='store_true', help="Run in debug mode (default: False)")

    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subparser for batch normalization of names
    parser_batch_normalization = subparsers.add_parser("normalize_names", help="Normalize image filenames to a consistent format.")
    parser_batch_normalization.add_argument("zip_directory", help="Path to the directory containing zip files")
    parser_batch_normalization.add_argument("output_directory", help="Path to the output directory for processed images")

    # Subparser for auto-rotation of images
    parser_auto_rotation = subparsers.add_parser("auto_rotate", help="Automatically rotate images to align the color checker in the left bottom corner.")
    parser_auto_rotation.add_argument("input_directory_for_rotation", help="Path to the input directory containing images")
    parser_auto_rotation.add_argument("output_directory_for_rotation", help="Path to the output directory for rotated images")
    parser_auto_rotation.add_argument("--check_sample", action='store_true', help="Check sample images for rulers presence before processing")

    # Subparser for simple flower crop
    parser_simple_crop = subparsers.add_parser("simple_flower_crop", help="Crop flower images and replace non-flower pixels with a specified background color.")
    parser_simple_crop.add_argument('input_dir_crop', help='Input directory with images')
    parser_simple_crop.add_argument('output_dir_crop', help='Output directory')
    parser_simple_crop.add_argument('--background', nargs=3, type=int, default=[0, 0, 0],
                        help='Background RGB color (default: 0 0 0 for black)')
    parser_simple_crop.add_argument('--size', nargs=2, type=int, default=[800, 800],
                        help='Target size width height (default: 800 800)')
    parser_simple_crop.add_argument('--no-preserve-aspect', action='store_true',
                        help='Do not preserve relative flower sizes (old resize behavior)')

# ------------------------------------------------------------------------------------------------------------------------------------------------

    ## Get the arguments from the command line
    args = parser.parse_args()

    debug = args.debug

    zip_directory = args.zip_directory
    output_directory = args.output_directory

    input_directory_rotation = args.input_directory_for_rotation
    output_directory_rotation = args.output_directory_for_rotation
    check_rulers_presence = args.check_sample

    input_directory_crop = args.input_dir_crop
    output_directory_crop = args.output_dir_crop
    background_color = args.background
    target_size = args.size
    preserve_aspect = not args.no_preserve_aspect

# ------------------------------------------------------------------------------------------------------------------------------------------------

    ## COMMAND 1: Batch normalization of names

    if args.command == "normalize_names":
        
        # Check if the output directory exists, if not, create it
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            logger.info(f"Created output directory: {output_directory}")
        
        logger.info("Normalizing names...")
        success = extract_and_process_batches(zip_directory, output_directory)

        if success:
            logger.info("✅ Script completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Script failed. Please check the errors above.")
            sys.exit(1)

        logger.info("Normalization completed.")

# ------------------------------------------------------------------------------------------------------------------------------------------------
    ## COMMAND 2: Auto-rotation of images

    elif args.command == "auto_rotate":

        if not os.path.exists(output_directory_rotation):
            os.makedirs(output_directory_rotation)
            logger.info(f"Created output directory for rotation: {output_directory_rotation}")

        print("=== Image Auto-Rotation Script ===")
        print("Target: Color checker in bottom-right position")
        print(f"Input: {input_directory_rotation}")
        print(f"Output: {output_directory_rotation}")

        # Check for rulers first
        check_rulers_presence(input_directory_rotation)

        # Preprocess images
        preprocess_images(input_directory_rotation, output_directory_rotation, check_sample=check_rulers_presence, debug_mode=debug)
        logger.info("Rotation completed.")

# ------------------------------------------------------------------------------------------------------------------------------------------------
    ## COMMAND 3: Simple flower crop
    elif args.command == "simple_flower_crop":

        input_path = Path(input_directory_crop)
        output_path = Path(output_directory_crop)
        output_path.mkdir(exist_ok=True)
        
        background_color = tuple(background_color)
        target_size = tuple(target_size)
        preserve_aspect = not preserve_aspect
        
        # Get all image files
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.JPG")) + \
                    list(input_path.glob("*.png")) + list(input_path.glob("*.PNG"))
        
        if not image_files:
            print(f"No images found in {input_directory_crop}")
            return
        
        print(f"Found {len(image_files)} images")
        print(f"Background color: RGB{background_color}")
        print(f"Target size: {target_size}")
        print(f"Preserve relative sizes: {preserve_aspect}")
        
        # Sort files
        image_files.sort()
        
        # Step 1: Select crop area on first image
        first_image = image_files[0]
        crop_coords = select_crop_area(first_image)
        
        if crop_coords is None:
            print("Crop selection cancelled")
            return
        
        # Step 2: Process all images
        print(f"\nProcessing {len(image_files)} images...")
        successful = 0
        
        for i, img_file in enumerate(image_files):
            try:
                # Load image
                img = cv2.imread(str(img_file))
                if img is None:
                    print(f"Could not load: {img_file.name}")
                    continue
                
                # Process image
                result = process_image(img, crop_coords, background_color, target_size, preserve_aspect)
                
                # Save result
                output_file = output_path / f"cropped_{img_file.name}"
                cv2.imwrite(str(output_file), result)
                
                successful += 1
                print(f"✓ ({i+1}/{len(image_files)}) {img_file.name}")
                
            except Exception as e:
                print(f"✗ Error processing {img_file.name}: {e}")
        
        print(f"\nCompleted: {successful}/{len(image_files)} images processed")
        print(f"Output saved in: {output_path}")

if __name__ == "__main__":
    main()