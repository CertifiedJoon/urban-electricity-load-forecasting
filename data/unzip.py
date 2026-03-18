import os
import sys
from zipfile import ZipFile
from tqdm.auto import tqdm
from shutil import copyfileobj
from tqdm.utils import CallbackIOWrapper

def unzip_with_progress(zip_file_name, destination_dir="."):
    """
    Extracts a zip file to a destination directory with a progress bar.

    Args:
        zip_file_name (str): The path to the zip file.
        destination_dir (str): The destination directory for extraction.
    """
    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    with ZipFile(zip_file_name, 'r') as zipf:
        # Get all members and calculate the total size for the progress bar
        members = zipf.infolist()
        total_size = sum(getattr(i, "file_size", 0) for i in members)

        with tqdm(
            desc="Extracting",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for member in members:
                # If it's a directory, just create it and continue
                if member.is_dir():
                    zipf.extract(member, destination_dir)
                    continue
                
                # Extract file with progress
                with zipf.open(member) as source, open(os.path.join(destination_dir, member.filename), "wb") as destination:
                    copyfileobj(CallbackIOWrapper(pbar.update, source), destination)
                

# Example Usage:
# Make sure you have a 'my_archive.zip' file in the same directory
# unzip_with_progress('my_archive.zip', 'extracted_content')
if __name__=="__main__":
  print(sys.argv)
  unzip_with_progress(sys.argv[1])