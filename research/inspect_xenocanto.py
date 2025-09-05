#!/usr/bin/env python3
"""
Deeper inspection of the XenoCanto dataset
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def inspect_xenocanto_dataset():
    """Get detailed info about the XenoCanto dataset"""
    
    try:
        import kaggle
        
        dataset_id = "imoore/xenocanto-bird-recordings-dataset"
        print(f"Detailed inspection of: {dataset_id}")
        print("=" * 60)
        
        # Try to get dataset metadata
        try:
            dataset_info = kaggle.api.dataset_view(dataset_id)
            print(f"Dataset Title: {dataset_info.title}")
            print(f"Dataset Size: {dataset_info.totalBytes} bytes")
            print(f"Download Count: {dataset_info.downloadCount}")
            print(f"Last Updated: {dataset_info.lastUpdated}")
            print(f"License: {dataset_info.licenseName}")
            print(f"Description: {dataset_info.description[:200]}...")
        except Exception as e:
            print(f"Could not get dataset metadata: {e}")
        
        # Try to list files with more detailed output
        print(f"\nFile listing attempt...")
        try:
            files_response = kaggle.api.dataset_list_files(dataset_id)
            
            # Check if it's a generator or has different structure
            print(f"Response type: {type(files_response)}")
            print(f"Response attributes: {dir(files_response)}")
            
            if hasattr(files_response, 'files'):
                files = files_response.files
                print(f"Files count: {len(files)}")
                
                for i, f in enumerate(files[:10]):  # Show first 10
                    print(f"  {i+1}. {f.name} ({f.size} bytes)")
            else:
                print("No 'files' attribute found")
                
        except Exception as e:
            print(f"Error listing files: {e}")
        
        # Try downloading to see if there's actual content
        print(f"\nTrying sample download...")
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                kaggle.api.dataset_download_files(
                    dataset_id, 
                    path=temp_dir, 
                    unzip=True
                )
                
                # Check what was downloaded
                temp_path = Path(temp_dir)
                downloaded_files = list(temp_path.rglob("*"))
                print(f"Downloaded {len(downloaded_files)} items:")
                
                for item in downloaded_files[:10]:
                    if item.is_file():
                        print(f"  📄 {item.name} ({item.stat().st_size} bytes)")
                    else:
                        print(f"  📁 {item.name}/")
                        
        except Exception as e:
            print(f"Download test failed: {e}")
    
    except ImportError:
        print("Kaggle not installed")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    inspect_xenocanto_dataset()
