import os
from PIL import Image

# Your dataset root folder
dataset_path = "data"

bad_files = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        file_path = os.path.join(root, file)

        try:
            # Try opening image
            with Image.open(file_path) as img:
                img.verify()  # verify integrity

        except Exception as e:
            print(f"❌ Removing corrupted file: {file_path}")
            bad_files.append(file_path)

            try:
                os.remove(file_path)
            except:
                print(f"⚠️ Could not delete: {file_path}")

print("\n✅ Cleaning complete!")
print(f"🗑️ Removed {len(bad_files)} corrupted files")