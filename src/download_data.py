import os
import requests
import zipfile
from tqdm import tqdm

DATA_URL = "http://mattmahoney.net/dc/text8.zip"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
FILE_PATH = os.path.join(DATA_DIR, "text8.zip")
EXTRACTED_PATH = os.path.join(DATA_DIR, "text8")

def download_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if os.path.exists(EXTRACTED_PATH):
        print(f"Data already exists at {EXTRACTED_PATH}")
        return

    if not os.path.exists(FILE_PATH):
        print(f"Downloading text8 dataset from {DATA_URL}...")
        response = requests.get(DATA_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(FILE_PATH, "wb") as file, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    else:
        print("Zip file already exists.")

    print("Extracting...")
    with zipfile.ZipFile(FILE_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    print(f"Data ready at {EXTRACTED_PATH}")

if __name__ == "__main__":
    download_data()
