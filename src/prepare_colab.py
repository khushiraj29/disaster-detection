import os
import zipfile

def prepare_colab_zip():
    zip_name = "disaster_project_colab.zip"
    files_to_zip = [
        "src/preprocessing.py",
        "src/model.py",
        "src/train.py",
        "src/__init__.py",
        "download_dataset.py",
        "requirements.txt"
    ]
    
    print(f"📦 Creating {zip_name} for Google Colab...")
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for file in files_to_zip:
            if os.path.exists(file):
                zipf.write(file)
            else:
                print(f"⚠️ Warning: {file} not found, skipping.")
                
    print(f"✅ Success! Upload '{zip_name}' to Colab and run '!unzip {zip_name}'")

if __name__ == "__main__":
    prepare_colab_zip()
