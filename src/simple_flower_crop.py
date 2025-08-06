#!/usr/bin/env python3
"""
Simple Flower Cropping Script
1. Manually select blue rectangle area on first image
2. Apply same crop to all images
3. Replace non-flower pixels with specified background color
4. Flower detection: anything that's not black/dark
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

# Global variables for rectangle selection
selecting = False
start_point = None
end_point = None
crop_rect = None

def mouse_callback(event, x, y, flags, param):
    """Mouse callback for rectangle selection"""
    global selecting, start_point, end_point, crop_rect
    
    img = param['img']
    
    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True
        start_point = (x, y)
    
    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        temp_img = img.copy()
        cv2.rectangle(temp_img, start_point, (x, y), (0, 255, 0), 2)
        cv2.imshow('Select Blue Rectangle Area', temp_img)
    
    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        end_point = (x, y)
        
        # Calculate rectangle coordinates
        x1, y1 = start_point
        x2, y2 = end_point
        
        crop_x = min(x1, x2)
        crop_y = min(y1, y2)
        crop_w = abs(x2 - x1)
        crop_h = abs(y2 - y1)
        
        crop_rect = (crop_x, crop_y, crop_w, crop_h)
        
        # Show final selection
        final_img = img.copy()
        cv2.rectangle(final_img, (crop_x, crop_y), (crop_x + crop_w, crop_y + crop_h), (0, 255, 0), 2)
        cv2.putText(final_img, 'Press SPACE to confirm', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Select Blue Rectangle Area', final_img)

def select_crop_area(image_path):
    """Select the blue rectangle area manually"""
    global crop_rect
    
    print(f"Select the blue rectangle area on: {image_path}")
    print("Click and drag to select the area, then press SPACE to confirm")
    
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not load image: {image_path}")
        return None
    
    # Resize for display if too large
    h, w = img.shape[:2]
    if max(h, w) > 1000:
        scale = 1000 / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        display_img = cv2.resize(img, (new_w, new_h))
        scale_factor = scale
    else:
        display_img = img.copy()
        scale_factor = 1.0
    
    cv2.namedWindow('Select Blue Rectangle Area')
    cv2.setMouseCallback('Select Blue Rectangle Area', mouse_callback, {'img': display_img})
    cv2.imshow('Select Blue Rectangle Area', display_img)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and crop_rect is not None:
            # Scale back to original size
            if scale_factor != 1.0:
                x, y, w, h = crop_rect
                x = int(x / scale_factor)
                y = int(y / scale_factor)
                w = int(w / scale_factor)
                h = int(h / scale_factor)
                crop_rect = (x, y, w, h)
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None
    
    cv2.destroyAllWindows()
    print(f"Selected area: x={crop_rect[0]}, y={crop_rect[1]}, w={crop_rect[2]}, h={crop_rect[3]}")
    return crop_rect

def detect_flower_mask(image, background_threshold=50):
    """
    Detect flower by finding non-black/dark pixels
    Filters out small dust particles - keeps only the largest connected component (the flower)
    """
    # Check if image is empty
    if image.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create mask: flower is anything brighter than threshold
    flower_mask = gray > background_threshold
    
    # Clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    flower_mask = cv2.morphologyEx(flower_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    flower_mask = cv2.morphologyEx(flower_mask, cv2.MORPH_OPEN, kernel)
    
    # Find connected components and keep only the largest one (the flower)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(flower_mask)
    
    if num_labels <= 1:  # No components found
        return flower_mask
    
    # Find the largest component (excluding background which is label 0)
    largest_label = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]
    
    for i in range(2, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = i
    
    # Create mask with only the largest component (the flower)
    clean_flower_mask = (labels == largest_label).astype(np.uint8)
    
    return clean_flower_mask

def process_image(image, crop_coords, background_color=(0, 0, 0), target_size=(800, 800), preserve_aspect=True):
    """
    Process a single image:
    1. Crop to blue rectangle area
    2. Detect flower (non-black pixels, largest component only)
    3. Replace background with specified color
    4. Optionally resize while preserving aspect ratio and relative sizes
    """
    x, y, w, h = crop_coords
    img_h, img_w = image.shape[:2]
    
    # Ensure crop coordinates are within image bounds
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    
    # Check if we have a valid crop area
    if x >= x2 or y >= y2:
        print(f"Warning: Invalid crop area for image {img_w}x{img_h} with coords ({x},{y},{w},{h})")
        # Create a black image of target size as fallback
        return np.full((target_size[1], target_size[0], 3), background_color, dtype=np.uint8)
    
    # Crop to selected area with bounds checking
    cropped = image[y:y2, x:x2]
    
    # Check if cropped image is empty
    if cropped.size == 0:
        print(f"Warning: Empty crop for image {img_w}x{img_h} with coords ({x},{y},{w},{h})")
        return np.full((target_size[1], target_size[0], 3), background_color, dtype=np.uint8)
    
    # Detect flower mask (only largest component, removes dust)
    flower_mask = detect_flower_mask(cropped)
    
    # Create result image with background color
    result = np.full(cropped.shape, background_color, dtype=np.uint8)
    
    # Copy flower pixels
    result[flower_mask > 0] = cropped[flower_mask > 0]
    
    if preserve_aspect:
        # Preserve relative dimensions by centering crop on target canvas
        target_w, target_h = target_size
        crop_h, crop_w = result.shape[:2]
        
        # Create canvas with background color
        canvas = np.full((target_h, target_w, 3), background_color, dtype=np.uint8)
        
        # Calculate scaling to fit crop in target size while maintaining aspect ratio
        scale_x = target_w / crop_w
        scale_y = target_h / crop_h
        scale = min(scale_x, scale_y)
        
        # Don't upscale, only downscale if necessary
        if scale < 1.0:
            new_w = int(crop_w * scale)
            new_h = int(crop_h * scale)
            result = cv2.resize(result, (new_w, new_h))
        else:
            new_w, new_h = crop_w, crop_h
        
        # Center the crop on the canvas
        start_x = (target_w - new_w) // 2
        start_y = (target_h - new_h) // 2
        
        # Place the flower on the canvas
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = result
        
        return canvas
    else:
        # Simple resize (old behavior)
        final_result = cv2.resize(result, target_size)
        return final_result

def main():
    parser = argparse.ArgumentParser(description='Simple flower cropping with manual rectangle selection')
    parser.add_argument('input_dir', help='Input directory with images')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--background', nargs=3, type=int, default=[0, 0, 0],
                        help='Background RGB color (default: 0 0 0 for black)')
    parser.add_argument('--size', nargs=2, type=int, default=[800, 800],
                        help='Target size width height (default: 800 800)')
    parser.add_argument('--no-preserve-aspect', action='store_true',
                        help='Do not preserve relative flower sizes (old resize behavior)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    background_color = tuple(args.background)
    target_size = tuple(args.size)
    preserve_aspect = not args.no_preserve_aspect
    
    # Get all image files
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.JPG")) + \
                  list(input_path.glob("*.png")) + list(input_path.glob("*.PNG"))
    
    if not image_files:
        print(f"No images found in {args.input_dir}")
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
