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

    # Subparser for batch normalization of names

    parser_batch_normalization = parser.add_subparsers(title = "normalize_names", dest="command", description="Normalize image filenames to a consistent format.")

    parser_batch_normalization.add_argument("zip_directory", help="Path to the directory containing zip files")
    parser_batch_normalization.add_argument("output_directory", help="Path to the output directory for processed images")

    # Subparser for auto-rotation of images
    parser_auto_rotation = parser.add_subparsers(title = "auto_rotate", dest="command", description="Automatically rotate images to align the color checker in the left bottom corner.")

    parser_auto_rotation.add_argument("input_directory_for_rotation", help="Path to the input directory containing images")
    parser_auto_rotation.add_argument("output_directory_for_rotation", help="Path to the output directory for rotated images")
    parser_auto_rotation.add_argument("--check_sample", action='store_true', help="Check sample images for rulers presence before processing")

# ------------------------------------------------------------------------------------------------------------------------------------------------

    ## Get the arguments from the command line
    args = parser.parse_args()

    debug = args.debug

    zip_directory = args.zip_directory
    output_directory = args.output_directory

    input_directory_rotation = args.input_directory_for_rotation
    output_directory_rotation = args.output_directory_for_rotation
    check_rulers_presence = args.check_sample

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

if __name__ == "__main__":
    main()