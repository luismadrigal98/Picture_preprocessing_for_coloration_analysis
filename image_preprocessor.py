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

    # Subparser for batch normalization of names

    parser_batch_normalization = parser.add_subparsers(title = "Label normalization", dest="command", description="Normalize image filenames to a consistent format.")

    parser_batch_normalization.add_argument("zip_directory", help="Path to the directory containing zip files")
    parser_batch_normalization.add_argument("output_directory", help="Path to the output directory for processed images")

    ## Get the arguments from the command line
    args = parser.parse_args()

    zip_directory = args.zip_directory
    output_directory = args.output_directory

    ## COMMAND 1: Batch normalization of names

    if args.command == "normalize_names":
        logger.info("Normalizing names...")
        success = extract_and_process_batches(zip_directory, output_directory)

        if success:
            logger.info("✅ Script completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Script failed. Please check the errors above.")
            sys.exit(1)

        logger.info("Normalization completed.")

if __name__ == "__main__":
    main()