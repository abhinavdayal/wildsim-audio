"""
Audio Dataset Manager for Wildlife Acoustic Simulation
Handles downloading and managing audio datasets from various sources
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import json
import hashlib
from dataclasses import dataclass
import shutil
from tqdm import tqdm
import requests
import zipfile
import tarfile
from models import DatasetConfig, SourceType
import librosa
import soundfile as sf

class DatasetDownloader:
    """Handles downloading datasets from various sources"""
    
    def __init__(self, cache_dir: str = "./audio_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_kaggle_dataset(self, dataset_id: str, cache_path: Path) -> bool:
        """Download dataset from Kaggle with improved progress tracking"""
        try:
            import kaggle
            import time
            import threading
            from concurrent.futures import ThreadPoolExecutor, TimeoutError
            
            print(f"📥 Starting download of Kaggle dataset: {dataset_id}")
            
            # Check if audio files already exist
            existing_audio_files = []
            for ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                existing_audio_files.extend(list(cache_path.rglob(f"*{ext}")))
            
            if existing_audio_files:
                print(f"✅ Found {len(existing_audio_files)} existing audio files")
                print("Skipping download - files already present")
                return True
            
            print(f"📁 Cache directory: {cache_path}")
            print("No audio files found - proceeding with download")
            
            # Download to temporary location
            temp_dir = cache_path / "temp"
            temp_dir.mkdir(exist_ok=True)
            
            # Function to run the download
            def do_download():
                try:
                    if "/" in dataset_id:
                        # Regular dataset
                        print(f"📦 Downloading regular dataset: {dataset_id}")
                        kaggle.api.dataset_download_files(
                            dataset_id, 
                            path=str(temp_dir), 
                            unzip=True
                        )
                    else:
                        # Competition dataset
                        print(f"🏆 Downloading competition dataset: {dataset_id}")
                        kaggle.api.competition_download_files(
                            dataset_id,
                            path=str(temp_dir),
                            unzip=True
                        )
                    return True
                except Exception as e:
                    print(f"❌ Download error: {e}")
                    return False
            
            # Run download with timeout and progress monitoring
            print("⏳ Starting download (this may take several minutes)...")
            start_time = time.time()
            
            # Use ThreadPoolExecutor for timeout control
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(do_download)
                
                # Monitor progress
                while not future.done():
                    elapsed = time.time() - start_time
                    print(f"📊 Download in progress... {elapsed:.0f}s elapsed")
                    
                    # Check if files are appearing
                    if temp_dir.exists():
                        temp_files = list(temp_dir.rglob("*"))
                        if temp_files:
                            total_size = sum(f.stat().st_size for f in temp_files if f.is_file())
                            print(f"📁 {len(temp_files)} files, {total_size/1024/1024:.1f}MB downloaded so far")
                    
                    time.sleep(10)  # Check every 10 seconds
                    
                    # Timeout after 5 minutes for initial testing
                    if elapsed > 300:
                        print("⏰ Download timeout (5 minutes) - cancelling")
                        future.cancel()
                        return False
                
                # Get result
                try:
                    success = future.result(timeout=30)
                    if not success:
                        return False
                except TimeoutError:
                    print("⏰ Download timed out")
                    return False
            
            print(f"📂 Checking downloaded files in {temp_dir}")
            temp_files = list(temp_dir.rglob("*"))
            print(f"Found {len(temp_files)} items in temp directory")
            
            if not temp_files:
                print("❌ No files were downloaded")
                return False
            
            # Move files to final location
            print("📦 Moving files to final location...")
            moved_count = 0
            
            for item in temp_dir.iterdir():
                try:
                    if item.is_dir():
                        dest = cache_path / item.name
                        if not dest.exists():
                            shutil.move(str(item), str(dest))
                            moved_count += 1
                            print(f"📁 Moved directory: {item.name}")
                    else:
                        dest = cache_path / item.name
                        if not dest.exists():
                            shutil.move(str(item), str(dest))
                            moved_count += 1
                            print(f"📄 Moved file: {item.name}")
                except Exception as e:
                    print(f"⚠️ Error moving {item.name}: {e}")
            
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                print("🧹 Cleaned up temp directory")
            except:
                pass
            
            # Final verification
            final_files = list(cache_path.rglob("*"))
            audio_files = [f for f in final_files if f.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg']]
            
            print(f"✅ Download completed!")
            print(f"📊 Total files: {len(final_files)}")
            print(f"🎵 Audio files: {len(audio_files)}")
            print(f"⏱️ Total time: {time.time() - start_time:.1f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {dataset_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def download_google_drive_folder(self, folder_id: str, cache_path: Path) -> bool:
        """Download folder from Google Drive (requires authentication)"""
        try:
            from googleapiclient.discovery import build
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            import pickle
            
            print(f"Downloading Google Drive folder: {folder_id}")
            
            # Google Drive API setup
            SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
            
            creds = None
            token_path = self.cache_dir / "drive_token.pickle"
            
            if token_path.exists():
                with open(token_path, 'rb') as token:
                    creds = pickle.load(token)
            
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    credentials_path = self.cache_dir / "drive_credentials.json"
                    if not credentials_path.exists():
                        print("❌ Google Drive credentials.json not found")
                        print("Please download from Google Cloud Console and place in cache directory")
                        return False
                        
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(credentials_path), SCOPES)
                    creds = flow.run_local_server(port=0)
                
                with open(token_path, 'wb') as token:
                    pickle.dump(creds, token)
            
            service = build('drive', 'v3', credentials=creds)
            
            # Download folder contents recursively
            self._download_drive_folder_recursive(service, folder_id, cache_path)
            
            print(f"✓ Successfully downloaded Google Drive folder")
            return True
            
        except ImportError:
            print("❌ Google Drive API not installed. Run: pip install google-api-python-client google-auth-oauthlib")
            return False
        except Exception as e:
            print(f"✗ Failed to download Google Drive folder: {e}")
            return False
    
    def _download_drive_folder_recursive(self, service, folder_id: str, local_path: Path):
        """Recursively download Google Drive folder contents"""
        
        # List files in folder
        query = f"'{folder_id}' in parents and trashed=false"
        results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
        files = results.get('files', [])
        
        for file_item in tqdm(files, desc="Downloading files"):
            file_name = file_item['name']
            file_id = file_item['id']
            mime_type = file_item['mimeType']
            
            if mime_type == 'application/vnd.google-apps.folder':
                # Recursive folder download
                subfolder_path = local_path / file_name
                subfolder_path.mkdir(exist_ok=True)
                self._download_drive_folder_recursive(service, file_id, subfolder_path)
            else:
                # Download file
                request = service.files().get_media(fileId=file_id)
                file_path = local_path / file_name
                
                with open(file_path, 'wb') as f:
                    downloader = MediaIoBaseDownload(f, request)
                    done = False
                    while done is False:
                        status, done = downloader.next_chunk()
    
    def download_url_dataset(self, url: str, cache_path: Path, extract: bool = True) -> bool:
        """Download dataset from URL"""
        try:
            print(f"Downloading from URL: {url}")
            
            # Download file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Determine filename
            if 'Content-Disposition' in response.headers:
                filename = response.headers['Content-Disposition'].split('filename=')[1].strip('"')
            else:
                filename = url.split('/')[-1]
            
            file_path = cache_path / filename
            
            # Download with progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
            
            # Extract if needed
            if extract and filename.endswith(('.zip', '.tar.gz', '.tgz')):
                self._extract_archive(file_path, cache_path)
                file_path.unlink()  # Remove archive after extraction
            
            print(f"✓ Successfully downloaded from URL")
            return True
            
        except Exception as e:
            print(f"✗ Failed to download from URL: {e}")
            return False
    
    def _extract_archive(self, archive_path: Path, extract_path: Path):
        """Extract compressed archive"""
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_path)

class AudioCatalog:
    """Catalogs and manages audio files with metadata"""
    
    def __init__(self, cache_dir: str = "./audio_cache"):
        self.cache_dir = Path(cache_dir)
        self.catalog_file = self.cache_dir / "audio_catalog.json"
        self.catalog = self._load_catalog()
    
    def _load_catalog(self) -> Dict:
        """Load existing catalog or create new one"""
        if self.catalog_file.exists():
            try:
                with open(self.catalog_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            "datasets": {},
            "files": {},
            "metadata": {
                "version": "1.0",
                "last_updated": None,
                "total_files": 0,
                "total_duration": 0.0
            }
        }
    
    def _save_catalog(self):
        """Save catalog to disk"""
        import datetime
        
        self.catalog["metadata"]["last_updated"] = datetime.datetime.now().isoformat()
        self.catalog["metadata"]["total_files"] = len(self.catalog["files"])
        
        with open(self.catalog_file, 'w') as f:
            json.dump(self.catalog, f, indent=2)
    
    def scan_dataset(self, dataset_config: DatasetConfig) -> Dict:
        """Scan dataset directory and catalog audio files"""
        
        print(f"Scanning dataset: {dataset_config.dataset_name}")
        
        dataset_path = Path(dataset_config.path_or_id)
        if not dataset_path.exists():
            print(f"❌ Dataset path not found: {dataset_path}")
            return {}
        
        audio_files = []
        file_info = {}
        
        # Find audio files
        for ext in dataset_config.audio_extensions:
            audio_files.extend(list(dataset_path.rglob(f"*{ext}")))
        
        print(f"Found {len(audio_files)} audio files")
        
        # Process each file
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            try:
                # Calculate file hash for deduplication
                file_hash = self._calculate_file_hash(audio_file)
                
                # Get audio metadata
                audio_info = self._get_audio_info(audio_file)
                
                # Store file information
                relative_path = audio_file.relative_to(dataset_path)
                file_info[str(relative_path)] = {
                    "absolute_path": str(audio_file),
                    "hash": file_hash,
                    "dataset": dataset_config.dataset_name,
                    "source_type": self._infer_source_type(audio_file),
                    **audio_info
                }
                
                # Add to global catalog
                self.catalog["files"][file_hash] = file_info[str(relative_path)]
                
            except Exception as e:
                print(f"Warning: Could not process {audio_file}: {e}")
        
        # Store dataset information
        self.catalog["datasets"][dataset_config.dataset_name] = {
            "config": dataset_config.dict(),
            "file_count": len(file_info),
            "scan_date": pd.Timestamp.now().isoformat()
        }
        
        self._save_catalog()
        
        print(f"✓ Cataloged {len(file_info)} files from {dataset_config.dataset_name}")
        return file_info
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        hasher = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        
        return hasher.hexdigest()[:16]  # Use first 16 chars
    
    def _get_audio_info(self, file_path: Path) -> Dict:
        """Extract audio file metadata"""
        try:
            # Use librosa for audio info
            y, sr = librosa.load(file_path, sr=None)
            duration = len(y) / sr
            
            # Get file stats
            stat = file_path.stat()
            
            return {
                "duration": duration,
                "sample_rate": sr,
                "channels": 1 if y.ndim == 1 else y.shape[0],
                "file_size": stat.st_size,
                "format": file_path.suffix.lower(),
                "peak_amplitude": float(np.max(np.abs(y))),
                "rms_level": float(np.sqrt(np.mean(y**2)))
            }
            
        except Exception as e:
            print(f"Warning: Could not analyze {file_path}: {e}")
            return {
                "duration": 0.0,
                "sample_rate": 16000,
                "channels": 1,
                "file_size": file_path.stat().st_size,
                "format": file_path.suffix.lower(),
                "peak_amplitude": 0.0,
                "rms_level": 0.0
            }
    
    def _infer_source_type(self, file_path: Path) -> str:
        """Infer source type from filename/path"""
        path_str = str(file_path).lower()
        
        # Keywords for different source types
        keywords = {
            SourceType.ELEPHANT: ['elephant', 'jumbo', 'tusker'],
            SourceType.BIRD: ['bird', 'avian', 'chirp', 'tweet', 'song'],
            SourceType.VEHICLE: ['car', 'truck', 'vehicle', 'engine', 'traffic'],
            SourceType.MONKEY: ['monkey', 'primate', 'ape', 'chimp'],
            SourceType.MACHINERY: ['machine', 'chainsaw', 'drill', 'construction'],
            SourceType.WATER: ['water', 'rain', 'river', 'stream'],
            SourceType.INSECT: ['insect', 'cricket', 'cicada', 'buzz']
        }
        
        for source_type, words in keywords.items():
            if any(word in path_str for word in words):
                return source_type.value
        
        return SourceType.UNKNOWN.value
    
    def find_audio_files(self, source_type: Optional[SourceType] = None, 
                        min_duration: float = 0.0,
                        max_duration: float = float('inf'),
                        dataset_name: Optional[str] = None) -> List[Dict]:
        """Find audio files matching criteria"""
        
        matches = []
        
        for file_hash, file_info in self.catalog["files"].items():
            # Filter by source type
            if source_type and file_info.get("source_type") != source_type.value:
                continue
            
            # Filter by duration
            duration = file_info.get("duration", 0.0)
            if duration < min_duration or duration > max_duration:
                continue
            
            # Filter by dataset
            if dataset_name and file_info.get("dataset") != dataset_name:
                continue
                
            matches.append(file_info)
        
        return matches
    
    def get_random_audio_file(self, source_type: Optional[SourceType] = None, 
                             min_duration: float = 0.0) -> Optional[str]:
        """Get random audio file matching criteria"""
        matches = self.find_audio_files(source_type, min_duration)
        
        if matches:
            import random
            selected = random.choice(matches)
            return selected.get("absolute_path")
        
        return None
    
    def process_audio_chunk(self, audio_files: list, config: DatasetConfig):
        """Process a chunk of audio files for Streamlit progress"""
        dataset_path = Path(config.cache_dir)
        
        for audio_file in audio_files:
            try:
                # Calculate file hash for deduplication
                file_hash = self._calculate_file_hash(audio_file)
                
                # Get audio metadata
                audio_info = self._get_audio_info(audio_file)
                
                # Store file information
                relative_path = audio_file.relative_to(dataset_path)
                file_info = {
                    "absolute_path": str(audio_file),
                    "hash": file_hash,
                    "dataset": config.dataset_name,
                    "source_type": self._infer_source_type(audio_file),
                    **audio_info
                }
                
                # Add to global catalog
                self.catalog["files"][file_hash] = file_info
                
            except Exception as e:
                print(f"Warning: Could not process {audio_file}: {e}")
    
    def finalize_dataset_catalog(self, config: DatasetConfig):
        """Finalize the dataset catalog after chunk processing"""
        # Count files for this dataset
        dataset_files = [f for f in self.catalog["files"].values() 
                        if f.get("dataset") == config.dataset_name]
        
        # Store dataset information
        self.catalog["datasets"][config.dataset_name] = {
            "config": config.dict(),
            "file_count": len(dataset_files),
            "scan_date": pd.Timestamp.now().isoformat()
        }
        
        self._save_catalog()
        print(f"✓ Cataloged {len(dataset_files)} files from {config.dataset_name}")

class SmartAudioSelector:
    """Intelligently selects appropriate audio files for simulation sources"""
    
    def __init__(self, dataset_manager):
        self.dataset_manager = dataset_manager
    
    def select_elephant_audio(self, context: str = "general") -> Optional[str]:
        """Select appropriate elephant audio based on context"""
        
        # Try to find elephant audio
        elephant_files = self.dataset_manager.catalog.find_audio_files(
            source_type=SourceType.ELEPHANT,
            min_duration=2.0
        )
        
        if elephant_files:
            import random
            selected = random.choice(elephant_files)
            return selected.get("absolute_path")
        
        return None

class DatasetManager:
    """Main dataset management class"""
    
    def __init__(self, cache_dir: str = "./audio_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.downloader = DatasetDownloader(str(self.cache_dir))
        self.catalog = AudioCatalog(str(self.cache_dir))
    
    def setup_dataset(self, config: DatasetConfig) -> bool:
        """Setup a dataset based on configuration"""
        
        print(f"Setting up dataset: {config.dataset_name}")
        
        success = False
        scan_path = None
        
        if config.source_type == "local":
            # Local dataset - just catalog existing files
            local_path = Path(config.path_or_id)
            if local_path.exists():
                success = True
                scan_path = local_path
                print(f"✓ Local dataset found at {local_path}")
            else:
                print(f"✗ Local path not found: {local_path}")
                
        else:
            # For non-local datasets, create cache directory
            dataset_path = Path(config.cache_dir)
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            if config.source_type == "kaggle":
                success = self.downloader.download_kaggle_dataset(
                    config.path_or_id, dataset_path
                )
                # For Kaggle, scan the cache directory where files were downloaded
                if success:
                    scan_path = dataset_path
                
            elif config.source_type == "google_drive":
                success = self.downloader.download_google_drive_folder(
                    config.path_or_id, dataset_path
                )
                if success:
                    scan_path = dataset_path
                
            elif config.source_type == "url":
                success = self.downloader.download_url_dataset(
                    config.path_or_id, dataset_path
                )
                if success:
                    scan_path = dataset_path
        
        if success and scan_path:
            # Create a modified config for scanning with the correct path
            scan_config = DatasetConfig(
                dataset_name=config.dataset_name,
                source_type=config.source_type,
                path_or_id=str(scan_path),  # Use the actual path where files are located
                audio_extensions=config.audio_extensions,
                metadata_file=config.metadata_file,
                cache_dir=config.cache_dir
            )
            
            # Catalog the dataset
            self.catalog.scan_dataset(scan_config)
            
        return success
    
    def setup_dataset_download_only(self, config: DatasetConfig) -> bool:
        """Setup a dataset (download only, no cataloging) for Streamlit progress"""
        
        print(f"Downloading dataset: {config.dataset_name}")
        
        dataset_path = Path(config.cache_dir)
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        success = False
        
        if config.source_type == "local":
            # Local dataset - just verify it exists
            local_path = Path(config.path_or_id)
            if local_path.exists():
                success = True
                print(f"✓ Local dataset found at {local_path}")
            else:
                print(f"✗ Local path not found: {local_path}")
                
        elif config.source_type == "kaggle":
            success = self.downloader.download_kaggle_dataset(
                config.path_or_id, dataset_path
            )
            
        elif config.source_type == "google_drive":
            success = self.downloader.download_google_drive_folder(
                config.path_or_id, dataset_path
            )
            
        elif config.source_type == "url":
            success = self.downloader.download_url_dataset(
                config.path_or_id, dataset_path
            )
        
        return success

    def get_dataset_summary(self) -> Dict:
        """Get summary of all datasets"""
        
        catalog_data = self.catalog.catalog
        
        summary = {
            "total_datasets": len(catalog_data.get("datasets", {})),
            "total_files": len(catalog_data.get("files", {})),
            "datasets": catalog_data.get("datasets", {}),
            "source_type_counts": {},
            "format_counts": {}
        }
        
        # Count by source type and format
        for file_info in catalog_data.get("files", {}).values():
            source_type = file_info.get("source_type", "unknown")
            summary["source_type_counts"][source_type] = summary["source_type_counts"].get(source_type, 0) + 1
            
            format_ext = file_info.get("format", "unknown")
            summary["format_counts"][format_ext] = summary["format_counts"].get(format_ext, 0) + 1
        
        return summary

if __name__ == "__main__":
    # Test dataset manager
    manager = DatasetManager()
    
    print("Dataset Manager Test")
    print(f"Cache directory: {manager.cache_dir}")
    
    summary = manager.get_dataset_summary()
    print(f"Summary: {summary}")
    
    print("✓ Dataset Manager working")