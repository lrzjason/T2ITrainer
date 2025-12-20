import os
import argparse
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.hub.api import HubApi

def download_model(model_id: str, local_dir: str, token: str = None):
    """
    Download a model from ModelScope Hub.

    Args:
        model_id (str): Model ID on ModelScope, e.g., 'qwen/Qwen2-7B'
        local_dir (str): Local directory to save the model
        token (str, optional): API token for private/authorized models
    """
    print(f"Downloading model '{model_id}' to '{local_dir}'...")

    # Set token if provided (for private models)
    if token:
        api = HubApi()
        api.login(token)
        print("✅ Logged in with provided token.")

    try:
        # Download model
        snapshot_download(
            model_id=model_id,
            cache_dir=local_dir,
            revision='master',  # or specify a tag/branch like 'v1.0'
        )
        print(f"✅ Model downloaded successfully to: {os.path.abspath(local_dir)}")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a model from ModelScope Hub.")
    parser.add_argument("--model_id", type=str, help="Model ID on ModelScope (e.g., 'qwen/Qwen2-7B')")
    parser.add_argument("--local_dir", type=str, help="Local directory to store the model")
    parser.add_argument("--token", type=str, help="ModelScope API token for private models (optional)")

    args = parser.parse_args()
    args.model_id = "Tongyi_Interaction_Lab/Z-Image-Turbo"
    args.local_dir = "./Z-Image-Turbo"
    args.token = "ms-fd39258e-99c3-40d2-95ae-4fda617b9d12"  # Replace with your token if needed

    # Create local directory if not exists
    os.makedirs(args.local_dir, exist_ok=True)

    download_model(
        model_id=args.model_id,
        local_dir=args.local_dir,
        token=args.token
    )