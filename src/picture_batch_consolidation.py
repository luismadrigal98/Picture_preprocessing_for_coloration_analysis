import os
import shutil
import zipfile
from pathlib import Path
import glob
import sys
import traceback

def extract_and_process_batches(zip_directory, output_directory):
    """
    Extract zip files, consolidate JPG images, and create master metadata file.
    
    Args:
        zip_directory: Path to directory containing zip files (Batch_1.zip, Batch_2.zip, etc.)
        output_directory: Path where consolidated files will be saved
        
    Returns:
        bool: True if processing completed successfully, False otherwise
    """
    
    # Validate inputs first
    if not validate_inputs(zip_directory, output_directory):
        return False
    
    # Create output directory
    output_path = Path(output_directory)
    output_path.mkdir(exist_ok=True)
    
    # Create subdirectories
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Initialize master metadata list
    master_metadata = []
    processed_files = set()  # Track processed files to avoid duplicates
    
    # Process each zip file
    zip_files = sorted(glob.glob(os.path.join(zip_directory, "*.zip")))
    
    for zip_path in zip_files:
        print(f"Processing {os.path.basename(zip_path)}...")
        
        # Extract zip to temporary directory
        temp_dir = output_path / "temp_extract"
        
        try:
            # Clean up any existing temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find the extracted batch directory
            batch_dirs = list(temp_dir.glob("Batch_*"))
            if not batch_dirs:
                print(f"Warning: No Batch_ directory found in {zip_path}")
                continue
                
            batch_dir = batch_dirs[0]  # Should be only one
            
            # Read metadata file
            metadata_file = batch_dir / f"{batch_dir.name}_metadata.txt"
            if not metadata_file.exists():
                print(f"Warning: Metadata file not found in {batch_dir}")
                continue
                
            with open(metadata_file, 'r') as f:
                plant_ids = [line.strip() for line in f.readlines() if line.strip()]
            
            # Get all JPG files sorted by name (both extensions handled together)
            jpg_files = sorted(list(batch_dir.glob("*.jpg")) + list(batch_dir.glob("*.JPG")))
            
            if len(jpg_files) != len(plant_ids) * 3:
                print(f"Warning: Expected {len(plant_ids)*3} JPG files, found {len(jpg_files)} in {batch_dir}")
                print(f"Plant IDs: {len(plant_ids)}, JPG files: {len(jpg_files)}")
            
            # Process images in groups of 3
            for i, plant_id in enumerate(plant_ids):
                for photo_num in range(1, 4):  # 3 photos per plant
                    jpg_index = i * 3 + (photo_num - 1)
                    
                    if jpg_index < len(jpg_files):
                        source_file = jpg_files[jpg_index]
                        
                        # Create new filename with batch info to avoid collisions: P{plant_id}_{photo_num}_{batch}.jpg
                        batch_name = batch_dir.name.replace("Batch_", "B")  # Shorter batch identifier
                        new_filename = f"P{plant_id}_{photo_num}_{batch_name}.jpg"
                        destination = images_dir / new_filename
                        
                        # Check for duplicates
                        if new_filename in processed_files:
                            print(f"  Warning: Duplicate filename {new_filename} - skipping")
                            continue
                        
                        # Copy file
                        shutil.copy2(source_file, destination)
                        processed_files.add(new_filename)
                        
                        # Add to master metadata
                        master_metadata.append({
                            'filename': new_filename,
                            'plant_id': plant_id,
                            'photo_number': photo_num,
                            'batch': batch_dir.name,
                            'original_filename': source_file.name
                        })
                        
                        print(f"  Copied: {source_file.name} -> {new_filename}")
                    else:
                        print(f"  Warning: Missing photo {photo_num} for plant {plant_id}")
            
            print(f"  Processed {len(plant_ids)} plants from {batch_dir.name}")
            
        except Exception as e:
            print(f"Error processing {zip_path}: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
        finally:
            # Clean up temporary directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    # Write master metadata files
    
    # 1. Simple plant ID list (for quick reference)
    plant_list_file = output_path / "master_plant_list.txt"
    unique_plants = sorted(set([entry['plant_id'] for entry in master_metadata]))
    with open(plant_list_file, 'w') as f:
        for plant_id in unique_plants:
            f.write(f"{plant_id}\n")
    
    # 2. Detailed metadata CSV
    detailed_metadata_file = output_path / "master_metadata.csv"
    with open(detailed_metadata_file, 'w') as f:
        f.write("filename,plant_id,photo_number,batch,original_filename\n")
        for entry in master_metadata:
            f.write(f"{entry['filename']},{entry['plant_id']},{entry['photo_number']},{entry['batch']},{entry['original_filename']}\n")
    
    # 3. Plant summary (for averaging later)
    plant_summary_file = output_path / "plant_summary.txt"
    with open(plant_summary_file, 'w') as f:
        f.write("# Plant ID -> Image files (for averaging Patternize results)\n")
        for plant_id in unique_plants:
            images = [entry['filename'] for entry in master_metadata if entry['plant_id'] == plant_id]
            f.write(f"{plant_id}: {', '.join(images)}\n")
    
    print(f"\n=== CONSOLIDATION COMPLETE ===")
    print(f"Total plants processed: {len(unique_plants)}")
    print(f"Total images copied: {len(master_metadata)}")
    print(f"Output directory: {output_path}")
    print(f"Images location: {images_dir}")
    print(f"Master metadata: {detailed_metadata_file}")
    print(f"Plant list: {plant_list_file}")
    print(f"Plant summary: {plant_summary_file}")
    
    return True

def validate_inputs(zip_directory, output_directory):
    """
    Validate input directories before processing.
    
    Args:
        zip_directory: Path to directory containing zip files
        output_directory: Path where consolidated files will be saved
        
    Returns:
        bool: True if inputs are valid, False otherwise
    """
    # Check if zip directory exists
    zip_path = Path(zip_directory)
    if not zip_path.exists():
        print(f"Error: Zip directory does not exist: {zip_directory}")
        return False
    
    if not zip_path.is_dir():
        print(f"Error: Zip directory path is not a directory: {zip_directory}")
        return False
    
    # Check if there are any zip files
    zip_files = list(zip_path.glob("*.zip"))
    if not zip_files:
        print(f"Error: No zip files found in directory: {zip_directory}")
        return False
    
    print(f"Found {len(zip_files)} zip files to process")
    
    # Check if output directory is writable
    output_path = Path(output_directory)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        # Test write permission
        test_file = output_path / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
    except Exception as e:
        print(f"Error: Cannot write to output directory {output_directory}: {str(e)}")
        return False
    
    return True

def preview_processing(zip_directory):
    """
    Preview what files would be processed without actually extracting or copying anything.
    
    Args:
        zip_directory: Path to directory containing zip files
    """
    zip_files = sorted(glob.glob(os.path.join(zip_directory, "*.zip")))
    
    print(f"\n=== PREVIEW MODE ===")
    print(f"Found {len(zip_files)} zip files to process:")
    
    total_plants = 0
    total_images = 0
    
    for zip_path in zip_files:
        print(f"\n📁 {os.path.basename(zip_path)}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                # Look for batch directories
                batch_dirs = [f for f in file_list if f.startswith("Batch_") and f.endswith("/")]
                
                if not batch_dirs:
                    print(f"   ⚠️  No Batch_ directory found")
                    continue
                
                batch_dir = batch_dirs[0].rstrip("/")
                print(f"   📂 Found batch directory: {batch_dir}")
                
                # Look for metadata file
                metadata_files = [f for f in file_list if f.endswith("_metadata.txt")]
                if metadata_files:
                    print(f"   📝 Metadata file: {metadata_files[0]}")
                else:
                    print(f"   ⚠️  No metadata file found")
                
                # Count JPG files
                jpg_files = [f for f in file_list if f.lower().endswith(('.jpg', '.jpeg'))]
                print(f"   🖼️  JPG files: {len(jpg_files)}")
                
                if len(jpg_files) % 3 == 0:
                    estimated_plants = len(jpg_files) // 3
                    print(f"   🌱 Estimated plants: {estimated_plants}")
                    total_plants += estimated_plants
                    total_images += len(jpg_files)
                else:
                    print(f"   ⚠️  JPG count ({len(jpg_files)}) not divisible by 3")
                    
        except Exception as e:
            print(f"   ❌ Error reading zip: {str(e)}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total estimated plants: {total_plants}")
    print(f"Total estimated images: {total_images}")
    print(f"Expected output files: {total_images} image files + 3 metadata files")