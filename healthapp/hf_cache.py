import os
import requests


def download_if_missing():
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)

    files = [
        (
            "cache/faiss.index",
            os.getenv("FAISS_INDEX_URL")
        ),
        (
            "cache/metadata.pkl",
            os.getenv("FAISS_METADATA_URL")
        )
    ]

    for file_path, url in files:

        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)

            if size_mb > 1:
                print(f"✓ Using cached file: {file_path}")
                continue

            os.remove(file_path)

        if not url:
            raise ValueError(
                f"Environment variable missing for {file_path}"
            )

        print(f"Downloading {file_path}...")

        response = requests.get(
            url,
            stream=True,
            timeout=300
        )
        response.raise_for_status()

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)

        print(f"✓ Downloaded {file_path}")