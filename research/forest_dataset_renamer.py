#!/usr/bin/env python3
"""
Forest Dataset File Renamer
Renames forest dataset audio files based on their class from metadata.csv
Example: If class is 'Fire', files become fire01.wav, fire02.wav, etc.
"""

import pandas as pd
import os
import shutil
from pathlib import Path
from collections import defaultdict

def rename_forest_dataset():
    """Rename forest dataset files based on metadata classes"""
    
    # Define paths
    metadata_file = "/Users/abhinav/antamaya/wildsim-audio/audio_cache/forest/Metadata-20220916T202011Z-001/Metadata/Metadata V1.0 FSC22.csv"
    audio_dir = "/Users/abhinav/antamaya/wildsim-audio/audio_cache/forest/Audio Wise V1.0-20220916T202003Z-001/Audio Wise V1.0"
    
    # Create backup directory for safety
    backup_dir = "/Users/abhinav/antamaya/wildsim-audio/audio_cache/forest/original_backup"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Read metadata CSV
    print("Reading metadata CSV...")
    try:
        df = pd.read_csv(metadata_file)
        print(f"Found {len(df)} rows in metadata")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False
    
    # Clean the data - remove invalid rows
    print("\nCleaning metadata...")
    # Remove header row and any malformed entries
    df_clean = df[df['Class Name'].str.contains(r'^[A-Za-z]', na=False)]
    df_clean = df_clean.dropna()
    
    print(f"After cleaning: {len(df_clean)} valid rows")
    
    # Get unique classes and their counts
    class_counts = df_clean['Class Name'].value_counts()
    print(f"\nFound {len(class_counts)} unique classes:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} files")
    
    # Create counters for each class
    class_counters = defaultdict(int)
    
    # Process each file
    print("\nProcessing files...")
    success_count = 0
    error_count = 0
    
    for idx, row in df_clean.iterrows():
        try:
            original_filename = row['Dataset File Name']
            class_name = row['Class Name']
            
            # Increment counter for this class
            class_counters[class_name] += 1
            
            # Create new filename: class name + zero-padded counter
            new_filename = f"{class_name.lower()}{class_counters[class_name]:02d}.wav"
            
            # Define file paths
            source_path = os.path.join(audio_dir, original_filename)
            dest_path = os.path.join(output_dir, new_filename)
            
            # Check if source file exists
            if os.path.exists(source_path):
                # Copy file with new name
                shutil.copy2(source_path, dest_path)
                print(f"  {original_filename} -> {new_filename}")
                success_count += 1
            else:
                print(f"  WARNING: Source file not found: {original_filename}")
                error_count += 1
                
        except Exception as e:
            print(f"  ERROR processing {row.get('Dataset File Name', 'unknown')}: {e}")
            error_count += 1
    
    print(f"\nRenaming complete!")
    print(f"Successfully renamed: {success_count} files")
    print(f"Errors: {error_count} files")
    print(f"Output directory: {output_dir}")
    
    # Show final class distribution
    print(f"\nFinal file distribution:")
    for class_name, count in class_counters.items():
        print(f"  {class_name}: {count} files")
    
    return success_count > 0

def verify_renamed_files():
    """Verify the renamed files"""
    output_dir = "/Users/abhinav/antamaya/wildsim-audio/audio_cache/forest/renamed_files"
    
    if not os.path.exists(output_dir):
        print("Renamed files directory doesn't exist!")
        return
    
    files = list(Path(output_dir).glob("*.wav"))
    print(f"\nVerification: Found {len(files)} renamed files")
    
    # Group by class
    class_files = defaultdict(list)
    for file in files:
        # Extract class name (everything before the number)
        name = file.stem
        class_name = ''.join([c for c in name if not c.isdigit()])
        class_files[class_name].append(file.name)
    
    for class_name, file_list in sorted(class_files.items()):
        print(f"  {class_name}: {len(file_list)} files")
        # Show first few files as examples
        for f in sorted(file_list)[:3]:
            print(f"    - {f}")
        if len(file_list) > 3:
            print(f"    ... and {len(file_list) - 3} more")

if __name__ == "__main__":
    print("🌲 Forest Dataset File Renamer")
    print("=" * 50)
    
    # Run the renaming process
    success = rename_forest_dataset()
    
    if success:
        print("\n" + "=" * 50)
        verify_renamed_files()
    else:
        print("❌ Renaming process failed!")
