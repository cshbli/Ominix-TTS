import os
from huggingface_hub import snapshot_download

def download_folder_from_repo(repo_id, folder_path, local_dir=None):
    """
    Download a specific folder from a Hugging Face repository.
    
    Args:
        repo_id (str): The ID of the repository (e.g., "username/repo-name")
        folder_path (str): The path to the folder within the repository
        local_dir (str, optional): Local directory where files will be downloaded
        
    Returns:
        str: Path to the downloaded folder
    """
    # Make sure folder_path ends with /* to download all files in the folder
    pattern = f"{folder_path}/*" if not folder_path.endswith("/*") else folder_path
    
    # Download only files matching the pattern
    downloaded_repo_path = snapshot_download(
        repo_id=repo_id,
        allow_patterns=pattern,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    
    # Return the path to the specific folder
    full_path = os.path.join(downloaded_repo_path, folder_path)
    return full_path
