import cv2
import numpy as np
import os
from pathlib import Path
import pandas as pd

def detect_color_checker_position(image):
    """
    Detect position of color checker card (not ruler!) in image
    Specifically looks for X-Rite ColorChecker-style cards with vibrant color patches
    Returns: position ('top_left', 'top_right', 'bottom_left', 'bottom_right')
    """
    h, w = image.shape[:2]
    corner_size = min(h, w) // 3  # Slightly larger region to catch color checkers better
    
    # Define corner regions
    corners = {
        'top_left': image[0:corner_size, 0:corner_size],
        'top_right': image[0:corner_size, w-corner_size:w],
        'bottom_left': image[h-corner_size:h, 0:corner_size],
        'bottom_right': image[h-corner_size:h, w-corner_size:w]
    }
    
    scores = {}
    
    for position, corner in corners.items():
        score = 0
        
        # Convert to different color spaces for analysis
        corner_hsv = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)
        corner_lab = cv2.cvtColor(corner, cv2.COLOR_BGR2LAB)
        
        # Method 1: High saturation patches (key indicator of color checker vs ruler)
        # Rulers have mostly grayscale squares, color checkers have vibrant colors
        saturation = corner_hsv[:,:,1]
        high_sat_pixels = np.sum(saturation > 100)  # Pixels with significant saturation
        sat_ratio = high_sat_pixels / saturation.size
        score += sat_ratio * 100  # Strong weight for colorful regions (increased from 50)
        
        # Method 2: Color diversity in LAB space (better for perceptual differences)
        # Real color checkers have distinct colors across the spectrum
        corner_small = cv2.resize(corner_lab, (24, 24))
        
        # Sample colors and calculate color diversity
        a_channel = corner_small[:,:,1]  # Green-Red axis
        b_channel = corner_small[:,:,2]  # Blue-Yellow axis
        
        # Look for wide spread in color space (indicates diverse colors)
        a_range = np.ptp(a_channel)  # Peak-to-peak (max - min)
        b_range = np.ptp(b_channel)
        color_spread = (a_range + b_range) / 2
        score += color_spread * 2  # Increased weight for color diversity
        
        # Method 3: Detect rectangular grid pattern (color checker specific)
        gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
        
        # Look for color checker grid (typically 6x4 or 4x6 patches)
        # Use template matching approach for regular grid
        h_corner, w_corner = corner.shape[:2]
        
        # Check for grid-like intensity variations
        # Downsample to roughly match expected grid size
        grid_size = 12  # Looking for ~6x4 or 4x6 pattern
        corner_grid = cv2.resize(gray, (grid_size, grid_size))
        
        # Calculate local variance (patches should have uniform color within, vary between)
        kernel = np.ones((2,2), np.float32) / 4
        blurred = cv2.filter2D(corner_grid, -1, kernel)
        local_var = np.var(corner_grid - blurred)
        score += local_var * 0.01  # Much smaller weight for grid pattern
        
        # Method 4: Exclude ruler-like patterns
        # Rulers are typically long and thin with mostly grayscale gradient
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for ruler-like shapes (high aspect ratio rectangles)
        ruler_penalty = 0
        for contour in contours:
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            if w_rect > 10 and h_rect > 10:  # Ignore tiny contours
                aspect_ratio = max(w_rect, h_rect) / min(w_rect, h_rect)
                area = w_rect * h_rect
                
                # Penalize long thin rectangles (ruler-like)
                if aspect_ratio > 5 and area > corner.size * 0.1:
                    ruler_penalty += aspect_ratio * 2
        
        score -= ruler_penalty
        
        # Method 5: Look for color checker specific features
        # Color checkers often have a white border or distinctive layout
        
        # Check for presence of both bright and dark patches
        brightness = np.mean(corner_lab[:,:,0])  # L channel
        bright_patches = np.sum(corner_lab[:,:,0] > brightness + 30)
        dark_patches = np.sum(corner_lab[:,:,0] < brightness - 30)
        
        if bright_patches > 0 and dark_patches > 0:
            score += 10  # Bonus for having both bright and dark areas
        
        # Method 6: Specific color presence check
        # Look for colors typical in color checkers (blues, reds, greens, etc.)
        hsv_flat = corner_hsv.reshape(-1, 3)
        
        # Count pixels in different hue ranges (only if saturated enough)
        saturated_pixels = hsv_flat[hsv_flat[:,1] > 80]  # Only consider saturated pixels
        
        if len(saturated_pixels) > 0:
            hues = saturated_pixels[:,0]
            
            # Define color ranges (in HSV hue)
            color_ranges = [
                (0, 20),     # Red
                (20, 40),    # Orange/Yellow  
                (40, 80),    # Green
                (80, 130),   # Blue/Cyan
                (130, 160),  # Purple/Magenta
                (160, 180)   # Red (wrap around)
            ]
            
            colors_present = 0
            for hue_min, hue_max in color_ranges:
                if np.any((hues >= hue_min) & (hues <= hue_max)):
                    colors_present += 1
            
            # Bonus for having multiple distinct colors
            score += colors_present * 5  # Increased weight for color variety
        
        scores[position] = max(0, score)  # Ensure non-negative scores
    
    # Return position with highest score
    if not scores or all(score <= 0 for score in scores.values()):
        # Fallback: assume bottom-right if no clear detection
        return 'bottom_right'
    
    return max(scores, key=scores.get)

def rotate_image_to_standard(image, checker_position):
    """
    Rotate image so color checker is in bottom-right position
    """
    rotation_angles = {
        'bottom_right': 0,      # no rotation needed - already in target position
        'bottom_left': -90,     # rotate 90° counterclockwise
        'top_right': 90,        # rotate 90° clockwise
        'top_left': 180         # rotate 180°
    }
    
    angle = rotation_angles.get(checker_position, 0)
    
    if angle == 0:
        return image, angle
    
    # Get image dimensions
    h, w = image.shape[:2]
    
    # For proper rotation without cropping
    if angle == 90:
        # Rotate 90° clockwise: use cv2.ROTATE_90_CLOCKWISE
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == -90:
        # Rotate 90° counterclockwise: use cv2.ROTATE_90_COUNTERCLOCKWISE
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        # Rotate 180°: use cv2.ROTATE_180
        rotated = cv2.rotate(image, cv2.ROTATE_180)
    else:
        rotated = image
    
    return rotated, angle

def preprocess_images(input_dir, output_dir, check_sample=True, debug_mode=False):
    """
    Preprocess all images: detect orientation, rotate to standard position (color checker in bottom-right)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Get all JPG files
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.JPG"))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    results = []
    
    if debug_mode:
        print("Running in debug mode - testing sample images first...")
        test_sample_images(input_dir, num_samples=3)
        proceed = input("Continue with full processing? (y/n): ")
        if proceed.lower() != 'y':
            return
    
    if check_sample:
        print("Checking sample of images first...")
        sample_files = image_files[:min(20, len(image_files))]  # Check first 20 images or all if fewer
        
        orientations = {}
        for img_file in sample_files:
            try:
                image = cv2.imread(str(img_file))
                if image is not None:
                    position = detect_color_checker_position(image)
                    orientations[position] = orientations.get(position, 0) + 1
                else:
                    print(f"Warning: Could not load {img_file}")
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
        
        print("Color checker positions in sample:")
        for pos, count in orientations.items():
            print(f"  {pos}: {count} images")
        
        proceed = input("Continue with full processing? (y/n): ")
        if proceed.lower() != 'y':
            return
    
    print(f"Processing {len(image_files)} images...")
    print("Target: Color checker in bottom-right position")
    
    failed_files = []
    
    for i, img_file in enumerate(image_files):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(image_files)} images ({i/len(image_files)*100:.1f}%)")
        
        try:
            # Load image
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"Warning: Could not load {img_file}")
                failed_files.append(str(img_file))
                continue
            
            # Check image dimensions to avoid issues
            h, w = image.shape[:2]
            if h < 100 or w < 100:
                print(f"Warning: Image too small {img_file}: {w}x{h}")
                failed_files.append(str(img_file))
                continue
            
            # Detect color checker position
            checker_position = detect_color_checker_position(image)
            
            # Rotate to standard orientation (color checker in bottom-right)
            rotated_image, rotation_angle = rotate_image_to_standard(image, checker_position)
            
            # Verify rotation didn't break the image
            if rotated_image is None or rotated_image.size == 0:
                print(f"Warning: Rotation failed for {img_file}")
                failed_files.append(str(img_file))
                continue
            
            # Save processed image
            output_file = output_path / img_file.name
            success = cv2.imwrite(str(output_file), rotated_image)
            
            if not success:
                print(f"Warning: Could not save {output_file}")
                failed_files.append(str(img_file))
                continue
            
            # Record processing info
            results.append({
                'filename': img_file.name,
                'original_checker_position': checker_position,
                'rotation_applied': rotation_angle,
                'processed': True,
                'original_size': f"{w}x{h}",
                'final_size': f"{rotated_image.shape[1]}x{rotated_image.shape[0]}"
            })
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            failed_files.append(str(img_file))
            results.append({
                'filename': img_file.name,
                'original_checker_position': 'unknown',
                'rotation_applied': 'failed',
                'processed': False,
                'error': str(e)
            })
    
    # Save processing log
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path / "preprocessing_log.csv", index=False)
    
    print(f"\nPreprocessing complete!")
    print(f"Processed images saved to: {output_path}")
    print(f"Processing log: {output_path / 'preprocessing_log.csv'}")
    
    # Summary statistics
    successful = len([r for r in results if r.get('processed', False)])
    print(f"\nSummary:")
    print(f"  Total images: {len(image_files)}")
    print(f"  Successfully processed: {successful}")
    print(f"  Failed: {len(failed_files)}")
    
    if results:
        processed_results = [r for r in results if r.get('processed', False)]
        if processed_results:
            results_df = pd.DataFrame(processed_results)
            rotation_counts = results_df['rotation_applied'].value_counts()
            print(f"  Rotation statistics:")
            for angle, count in rotation_counts.items():
                direction = {0: "none", 90: "90° CW", -90: "90° CCW", 180: "180°"}.get(angle, f"{angle}°")
                print(f"    {direction}: {count} images")
    
    if failed_files:
        print(f"\nFailed files:")
        for failed in failed_files[:10]:  # Show first 10
            print(f"  {failed}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")

def check_rulers_presence(input_dir):
    """
    Check sample of images for ruler presence (vs color checker) to assess detection accuracy
    """
    input_path = Path(input_dir)
    image_files = list(input_path.glob("*.jpg"))[:50]  # Check 50 images
    
    print("Checking for rulers vs color checkers in sample images...")
    ruler_detected = 0
    color_checker_detected = 0
    
    for img_file in image_files:
        image = cv2.imread(str(img_file))
        if image is None:
            continue
            
        # Check each corner for ruler vs color checker characteristics
        h, w = image.shape[:2]
        corner_size = min(h, w) // 4
        
        corners = {
            'top_left': image[0:corner_size, 0:corner_size],
            'top_right': image[0:corner_size, w-corner_size:w],
            'bottom_left': image[h-corner_size:h, 0:corner_size],
            'bottom_right': image[h-corner_size:h, w-corner_size:w]
        }
        
        has_ruler = False
        has_color_checker = False
        
        for corner in corners.values():
            # Convert to HSV for better color analysis
            corner_hsv = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)
            saturation = corner_hsv[:,:,1]
            
            # Ruler detection: mostly grayscale with some colored squares
            low_sat_ratio = np.sum(saturation < 50) / saturation.size
            
            # Look for ruler-like shapes (long rectangles)
            gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            ruler_shapes = 0
            for contour in contours:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                aspect_ratio = max(w_rect, h_rect) / min(w_rect, h_rect)
                area = w_rect * h_rect
                
                # Ruler characteristics: high aspect ratio, moderate size, mostly grayscale
                if aspect_ratio > 8 and area > 1000 and low_sat_ratio > 0.7:
                    ruler_shapes += 1
            
            if ruler_shapes > 0:
                has_ruler = True
            
            # Color checker detection: high saturation, diverse colors
            high_sat_ratio = np.sum(saturation > 100) / saturation.size
            if high_sat_ratio > 0.1:  # At least 10% highly saturated pixels
                has_color_checker = True
        
        if has_ruler:
            ruler_detected += 1
        if has_color_checker:
            color_checker_detected += 1
    
    print(f"Rulers detected in {ruler_detected}/{len(image_files)} sample images")
    print(f"Color checkers detected in {color_checker_detected}/{len(image_files)} sample images")
    
    if ruler_detected > color_checker_detected:
        print("⚠️  Warning: More rulers than color checkers detected.")
        print("   The improved detection should now focus on actual color checkers.")
    else:
        print("✓ Good: More color checkers than rulers detected.")
    
    print("Note: Updated detection now specifically targets colorful checker cards, not rulers")

def debug_color_checker_detection(image_path, save_debug=False):
    """
    Debug function to visualize color checker detection with detailed analysis
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"\nDebugging: {image_path.name}")
    print(f"Image dimensions: {image.shape}")
    
    # Calculate scores step by step for each corner
    h, w = image.shape[:2]
    corner_size = min(h, w) // 3
    
    corners = {
        'top_left': image[0:corner_size, 0:corner_size],
        'top_right': image[0:corner_size, w-corner_size:w],
        'bottom_left': image[h-corner_size:h, 0:corner_size],
        'bottom_right': image[h-corner_size:h, w-corner_size:w]
    }
    
    scores = {}
    print("\nDetailed scoring analysis:")
    
    for position, corner in corners.items():
        score = 0
        
        # Convert to different color spaces for analysis
        corner_hsv = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)
        corner_lab = cv2.cvtColor(corner, cv2.COLOR_BGR2LAB)
        
        print(f"\n=== {position.upper()} ===")
        
        # Method 1: High saturation patches
        saturation = corner_hsv[:,:,1]
        high_sat_pixels = np.sum(saturation > 100)
        sat_ratio = high_sat_pixels / saturation.size
        sat_score = sat_ratio * 100  # Increased from 50
        score += sat_score
        print(f"1. Saturation score: {sat_score:.2f} (ratio: {sat_ratio:.3f})")
        
        # Method 2: Color diversity in LAB space
        corner_small = cv2.resize(corner_lab, (24, 24))
        a_channel = corner_small[:,:,1]
        b_channel = corner_small[:,:,2]
        a_range = np.ptp(a_channel)
        b_range = np.ptp(b_channel)
        color_spread = (a_range + b_range) / 2
        spread_score = color_spread * 2  # Increased from /5
        score += spread_score
        print(f"2. Color spread score: {spread_score:.2f} (spread: {color_spread:.1f})")
        
        # Method 3: Grid pattern detection
        gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
        grid_size = 12
        corner_grid = cv2.resize(gray, (grid_size, grid_size))
        kernel = np.ones((2,2), np.float32) / 4
        blurred = cv2.filter2D(corner_grid, -1, kernel)
        local_var = np.var(corner_grid - blurred)
        grid_score = local_var * 0.01  # Much smaller weight to prevent domination
        score += grid_score
        print(f"3. Grid pattern score: {grid_score:.2f} (variance: {local_var:.1f})")
        
        # Method 4: Ruler penalty
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ruler_penalty = 0
        for contour in contours:
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            if w_rect > 10 and h_rect > 10:
                aspect_ratio = max(w_rect, h_rect) / min(w_rect, h_rect)
                area = w_rect * h_rect
                
                if aspect_ratio > 5 and area > corner.size * 0.1:
                    ruler_penalty += aspect_ratio * 2
        
        score -= ruler_penalty
        print(f"4. Ruler penalty: -{ruler_penalty:.2f}")
        
        # Method 5: Bright/dark patch bonus
        brightness = np.mean(corner_lab[:,:,0])
        bright_patches = np.sum(corner_lab[:,:,0] > brightness + 30)
        dark_patches = np.sum(corner_lab[:,:,0] < brightness - 30)
        
        brightness_bonus = 10 if bright_patches > 0 and dark_patches > 0 else 0
        score += brightness_bonus
        print(f"5. Brightness variety bonus: {brightness_bonus}")
        
        # Method 6: Color presence check
        hsv_flat = corner_hsv.reshape(-1, 3)
        saturated_pixels = hsv_flat[hsv_flat[:,1] > 80]
        
        color_bonus = 0
        if len(saturated_pixels) > 0:
            hues = saturated_pixels[:,0]
            
            color_ranges = [
                (0, 20), (20, 40), (40, 80), (80, 130), (130, 160), (160, 180)
            ]
            
            colors_present = 0
            for hue_min, hue_max in color_ranges:
                if np.any((hues >= hue_min) & (hues <= hue_max)):
                    colors_present += 1
            
            color_bonus = colors_present * 5  # Increased from 3
        
        score += color_bonus
        print(f"6. Color variety bonus: {color_bonus} ({colors_present if len(saturated_pixels) > 0 else 0} colors)")
        
        final_score = max(0, score)
        scores[position] = final_score
        print(f"TOTAL SCORE: {final_score:.2f}")
    
    # Find winner
    best_position = max(scores, key=scores.get) if scores else 'bottom_right'
    print(f"\n🏆 WINNER: {best_position} (score: {scores[best_position]:.2f})")
    
    # Show all scores ranked
    print("\nAll scores ranked:")
    for pos, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pos}: {score:.2f}")
    
    # Detect color checker
    position = detect_color_checker_position(image)
    print(f"\nActual detection result: {position}")
    
    # Draw visualization
    debug_image = image.copy()
    
    # Draw corner regions with updated size
    corners_coords = {
        'top_left': (0, 0, corner_size, corner_size),
        'top_right': (w-corner_size, 0, w, corner_size),
        'bottom_left': (0, h-corner_size, corner_size, h),
        'bottom_right': (w-corner_size, h-corner_size, w, h)
    }
    
    for pos, (x1, y1, x2, y2) in corners_coords.items():
        color = (0, 255, 0) if pos == position else (0, 0, 255)  # Green for detected, red for others
        thickness = 3 if pos == position else 1
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, thickness)
        
        # Add text label with score
        score_text = f"{pos.replace('_', ' ')}: {scores[pos]:.1f}"
        cv2.putText(debug_image, score_text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Add detection result text
    cv2.putText(debug_image, f"Color Checker: {position}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if save_debug:
        debug_path = image_path.parent / f"debug_{image_path.name}"
        cv2.imwrite(str(debug_path), debug_image)
        print(f"Debug image saved: {debug_path}")
    
    return position, debug_image

def test_sample_images(input_dir, num_samples=5):
    """
    Test detection on a few sample images for verification
    """
    input_path = Path(input_dir)
    image_files = list(input_path.glob("*.jpg"))[:num_samples]
    
    print(f"Testing color checker detection on {len(image_files)} sample images...")
    
    for img_file in image_files:
        position, debug_img = debug_color_checker_detection(img_file, save_debug=True)
        
        # Show what rotation would be applied
        rotation_angles = {
            'bottom_right': 0, 'bottom_left': -90, 'top_right': 90, 'top_left': 180
        }
        angle = rotation_angles.get(position, 0)
        print(f"  Would rotate {angle}° to move checker to bottom-right")
        print("-" * 50)