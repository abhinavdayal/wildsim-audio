"""
Authentication helper functions for dataset sources
"""

import streamlit as st
import os
import json
from pathlib import Path

def check_kaggle_auth():
    """Check if Kaggle API is configured"""
    try:
        import kaggle
        # Try to get user info to verify authentication
        user_info = kaggle.api.get_config_value('username')
        return {
            'authenticated': True,
            'username': user_info
        }
    except Exception as e:
        return {
            'authenticated': False,
            'error': str(e)
        }

def setup_kaggle_credentials(uploaded_file):
    """Setup Kaggle credentials from uploaded file"""
    try:
        # Read the uploaded file
        credentials = json.loads(uploaded_file.read())
        
        # Create .kaggle directory
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_dir.mkdir(exist_ok=True)
        
        # Write credentials file
        with open(kaggle_dir / 'kaggle.json', 'w') as f:
            json.dump(credentials, f)
        
        # Set proper permissions (readable only by user)
        os.chmod(kaggle_dir / 'kaggle.json', 0o600)
        
        return True
    except Exception as e:
        st.error(f"Error setting up Kaggle credentials: {e}")
        return False

def check_gdrive_auth():
    """Check if Google Drive API is configured"""
    try:
        cache_dir = Path('./audio_cache')
        
        # Check for credentials file
        creds_file = cache_dir / 'drive_credentials.json'
        token_file = cache_dir / 'drive_token.pickle'
        
        return {
            'authenticated': creds_file.exists() or token_file.exists(),
            'has_credentials': creds_file.exists(),
            'has_token': token_file.exists()
        }
    except Exception as e:
        return {
            'authenticated': False,
            'error': str(e)
        }

def setup_gdrive_credentials(uploaded_file):
    """Setup Google Drive credentials from uploaded file"""
    try:
        # Read the uploaded file
        credentials = json.loads(uploaded_file.read())
        
        # Create cache directory
        cache_dir = Path('./audio_cache')
        cache_dir.mkdir(exist_ok=True)
        
        # Write credentials file
        with open(cache_dir / 'drive_credentials.json', 'w') as f:
            json.dump(credentials, f)
        
        return True
    except Exception as e:
        st.error(f"Error setting up Google Drive credentials: {e}")
        return False

def get_kaggle_dataset_info(dataset_id, dataset_type):
    """Get information about a Kaggle dataset"""
    try:
        import kaggle
        
        if dataset_type == "Dataset":
            # Get file list (this works and gives us some info)
            files_response = kaggle.api.dataset_list_files(dataset_id)
            files = files_response.files if hasattr(files_response, 'files') else []
            file_count = len(files)
            
            # Calculate total size if available
            total_size = 0
            for f in files:
                if hasattr(f, 'size') and f.size:
                    total_size += f.size
            
            return {
                'title': dataset_id.replace('/', ' - '),  # Use dataset_id as title
                'description': f'Dataset with {file_count} files',
                'files': file_count,
                'size_bytes': total_size,
                'size_mb': f"{total_size / (1024*1024):.1f} MB" if total_size > 0 else "Unknown",
                'url': f"https://www.kaggle.com/datasets/{dataset_id}",
                'id': dataset_id,
                'file_names': [f.name for f in files[:5]]  # Show first 5 files
            }
        else:
            # Competition - get competition file list
            files_response = kaggle.api.competition_list_files(dataset_id)
            files = files_response.files if hasattr(files_response, 'files') else []
            file_count = len(files)
            
            return {
                'title': f"Competition: {dataset_id}",
                'description': f'Competition with {file_count} files',
                'files': file_count,
                'url': f"https://www.kaggle.com/c/{dataset_id}",
                'id': dataset_id,
                'type': 'Competition',
                'file_names': [f.name for f in files[:5]]  # Show first 5 files
            }
    except Exception as e:
        # More detailed error handling
        error_msg = str(e)
        if "404" in error_msg:
            st.error(f"Dataset '{dataset_id}' not found. Please check the dataset ID.")
        elif "403" in error_msg:
            st.error(f"Access denied to dataset '{dataset_id}'. Make sure it's public or you have access.")
        else:
            st.error(f"Error fetching dataset info: {error_msg}")
        return None

def get_gdrive_folder_info(folder_id):
    """Get information about a Google Drive folder"""
    try:
        # For now, provide a simplified implementation that doesn't require OAuth flow
        # in the Streamlit environment. This would require manual OAuth setup.
        st.warning("📁 Google Drive folder preview requires OAuth authentication flow which is complex in web environments.")
        st.info(f"💡 To verify your folder, visit: https://drive.google.com/drive/folders/{folder_id}")
        
        # Return a placeholder response
        return {
            'name': f"Folder {folder_id}",
            'files': [{'name': 'OAuth setup required for file listing', 'size': 'N/A', 'type': 'info'}],
            'note': 'File listing requires completing OAuth flow'
        }
        
        # TODO: For production, implement proper OAuth flow with Streamlit secrets
        # or use service account authentication
        
    except Exception as e:
        st.error(f"Error accessing Google Drive: {e}")
        return None

def test_kaggle_dataset_access(dataset_id, dataset_type="Dataset"):
    """Test if a Kaggle dataset is accessible"""
    try:
        import kaggle
        
        print(f"Testing access to: {dataset_id}")
        
        if dataset_type == "Dataset":
            # Try to list files to test access
            files_response = kaggle.api.dataset_list_files(dataset_id)
            files = files_response.files if hasattr(files_response, 'files') else []
            
            return {
                'accessible': True,
                'file_count': len(files),
                'sample_files': [f.name for f in files[:3]],
                'dataset_url': f"https://www.kaggle.com/datasets/{dataset_id}"
            }
        else:
            # Competition dataset
            files_response = kaggle.api.competition_list_files(dataset_id)
            files = files_response.files if hasattr(files_response, 'files') else []
            
            return {
                'accessible': True,
                'file_count': len(files),
                'sample_files': [f.name for f in files[:3]],
                'dataset_url': f"https://www.kaggle.com/c/{dataset_id}"
            }
            
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            return {'accessible': False, 'error': 'Dataset not found', 'suggestion': 'Check dataset ID format'}
        elif "403" in error_msg:
            return {'accessible': False, 'error': 'Access denied', 'suggestion': 'Dataset may be private or requires permission'}
        else:
            return {'accessible': False, 'error': error_msg, 'suggestion': 'Check authentication and internet connection'}

def test_url_accessibility(url):
    """Test if a URL is accessible"""
    try:
        import requests
        response = requests.head(url, timeout=10)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error testing URL: {e}")
        return False