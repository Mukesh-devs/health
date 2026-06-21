import os
import requests


def download_if_missing():
    os.makedirs("cache", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)

    files = [
        (
            "cache/faiss.index",
            os.getenv("FAISS_INDEX_URL")
        ),
        (
            "cache/metadata.pkl",
            os.getenv("FAISS_METADATA_URL")
        ),
        (
            "dataset/neo4j_node.csv",
            os.getenv("KG_NODE_URL")
        ),
        (
            "dataset/neo4j_rel.csv",
            os.getenv("KG_REL_URL")
        )
    ]

    for file_path, url in files:

        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)

            if size_mb > 1:
                print(f"✓ Using cached file: {file_path} ({size_mb:.2f} MB)")
                continue

            print(f"⚠ Corrupted file detected: {file_path}")
            os.remove(file_path)

        if not url:
            raise ValueError(
                f"Environment variable missing for {file_path}"
            )

        print(f"Downloading {file_path}...")

        response = requests.get(
            url,
            stream=True,
            timeout=600
        )
        response.raise_for_status()

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"✓ Downloaded {file_path} ({size_mb:.2f} MB)")

    print("✓ All required files are available")