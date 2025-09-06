#!/usr/bin/env python3
"""
Wild Animals Dataset Renamer
Renames wild animals audio files to consistent naming pattern
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import pandas as pd

def rename_wild_animals_dataset():
    """Rename wild animals dataset files to consistent pattern"""
    
    # Base path for wild animals dataset
    base_path = Path("./audio_cache/wild_animals/Animal-Soundprepros")
    
    if not base_path.exists():
        print(f"❌ Wild animals dataset not found at {base_path}")
        return False
    
    print(f"🔍 Processing wild animals dataset at {base_path}")
    
    # Mapping from folder names to standard animal names (lowercase)
    folder_to_animal = {
        'Aslan': 'lion',        # Aslan is Turkish for lion
        'Bear': 'bear',
        'Cat': 'cat',
        'Chicken': 'chicken',
        'Cow': 'cow',
        'Dog': 'dog',
        'Dolphin': 'dolphin',
        'Donkey': 'donkey',
        'Elephant': 'elephant',
        'Frog': 'frog',
        'Horse': 'horse',
        'Monkey': 'monkey',
        'Sheep': 'sheep'
    }
    
    # Statistics tracking
    total_files = 0
    renamed_files = 0
    errors = []
    rename_summary = {}
    
    # Process each folder
    for folder_name, animal_name in folder_to_animal.items():
        folder_path = base_path / folder_name
        
        if not folder_path.exists():
            print(f"⚠️ Folder not found: {folder_name}")
            continue
        
        print(f"\n📁 Processing {folder_name} → {animal_name}")
        
        # Get all audio files in the folder
        audio_files = list(folder_path.glob("*.wav"))
        audio_files.sort()  # Sort for consistent numbering
        
        if not audio_files:
            print(f"   No audio files found in {folder_name}")
            continue
        
        print(f"   Found {len(audio_files)} files")
        total_files += len(audio_files)
        
        # Counter for consistent numbering
        counter = 1
        folder_renames = 0
        
        for old_file in audio_files:
            try:
                # Create new filename: animal_name + zero-padded counter
                new_filename = f"{animal_name}{counter:02d}.wav"
                new_path = old_file.parent / new_filename
                
                # Check if file already has the correct name
                if old_file.name == new_filename:
                    print(f"   ✓ {old_file.name} already correctly named")
                    counter += 1
                    continue
                
                # Check if target filename already exists
                if new_path.exists() and new_path != old_file:
                    print(f"   ⚠️ Target file {new_filename} already exists, skipping {old_file.name}")
                    counter += 1
                    continue
                
                # Rename the file
                old_file.rename(new_path)
                print(f"   ✓ {old_file.name} → {new_filename}")
                renamed_files += 1
                folder_renames += 1
                counter += 1
                
            except Exception as e:
                error_msg = f"Failed to rename {old_file.name}: {e}"
                errors.append(error_msg)
                print(f"   ❌ {error_msg}")
                counter += 1
        
        rename_summary[animal_name] = {
            'folder': folder_name,
            'total_files': len(audio_files),
            'renamed': folder_renames,
            'already_correct': len(audio_files) - folder_renames - len([e for e in errors if folder_name in e])
        }
        
        print(f"   📊 {folder_renames} files renamed, {len(audio_files) - folder_renames} already correct")
    
    # Create summary report
    print(f"\n🎯 Wild Animals Dataset Renaming Summary")
    print(f"=" * 50)
    print(f"Total files found: {total_files}")
    print(f"Files renamed: {renamed_files}")
    print(f"Errors: {len(errors)}")
    print(f"Success rate: {((total_files - len(errors)) / total_files * 100):.1f}%" if total_files > 0 else "N/A")
    
    # Detailed summary by animal
    print(f"\n📋 Detailed Summary by Animal:")
    for animal, stats in rename_summary.items():
        print(f"  {animal.capitalize():12} | {stats['total_files']:3d} files | {stats['renamed']:3d} renamed | {stats['already_correct']:3d} already correct")
    
    if errors:
        print(f"\n❌ Errors encountered:")
        for error in errors:
            print(f"   {error}")
    
    # Save summary to file
    summary_data = []
    for animal, stats in rename_summary.items():
        summary_data.append({
            'animal': animal,
            'folder_name': stats['folder'],
            'total_files': stats['total_files'],
            'renamed': stats['renamed'],
            'already_correct': stats['already_correct']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = Path("wild_animals_rename_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"\n💾 Summary saved to {summary_file}")
    
    return len(errors) == 0

def verify_wild_animals_renaming():
    """Verify the renaming was successful"""
    
    base_path = Path("./audio_cache/wild_animals/Animal-Soundprepros")
    
    if not base_path.exists():
        print("❌ Wild animals dataset not found")
        return False
    
    print("\n🔍 Verifying wild animals dataset renaming...")
    
    expected_animals = ['lion', 'bear', 'cat', 'chicken', 'cow', 'dog', 'dolphin', 'donkey', 'elephant', 'frog', 'horse', 'monkey', 'sheep']
    
    all_correct = True
    
    for folder in base_path.iterdir():
        if folder.is_dir():
            files = list(folder.glob("*.wav"))
            
            # Check if files follow the expected pattern
            pattern_files = []
            for animal in expected_animals:
                pattern_files.extend(list(folder.glob(f"{animal}*.wav")))
            
            if pattern_files:
                animal_type = pattern_files[0].stem.rstrip('0123456789')
                print(f"📁 {folder.name}: {len(pattern_files)} {animal_type} files")
                
                # Verify sequential numbering
                numbers = []
                for file in pattern_files:
                    # Extract number from filename like "elephant01.wav"
                    number_part = file.stem[len(animal_type):]
                    if number_part.isdigit():
                        numbers.append(int(number_part))
                
                numbers.sort()
                expected_numbers = list(range(1, len(numbers) + 1))
                
                if numbers == expected_numbers:
                    print(f"   ✅ Sequential numbering correct (1-{len(numbers)})")
                else:
                    print(f"   ❌ Sequential numbering incorrect")
                    print(f"      Expected: {expected_numbers}")
                    print(f"      Found: {numbers}")
                    all_correct = False
            else:
                print(f"📁 {folder.name}: No files found with expected pattern")
                all_correct = False
    
    return all_correct

if __name__ == "__main__":
    print("🚀 Wild Animals Dataset Renamer")
    print("=" * 40)
    
    # Rename files
    success = rename_wild_animals_dataset()
    
    if success:
        print("\n✅ Renaming completed successfully!")
        
        # Verify the renaming
        if verify_wild_animals_renaming():
            print("✅ Verification passed - all files correctly named!")
        else:
            print("⚠️ Verification found some issues")
    else:
        print("\n❌ Renaming completed with errors")
    
    print("\n🎯 Wild animals dataset processing complete!")
