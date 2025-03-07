import kagglehub
import os
import zipfile
import shutil

def download_and_extract_dataset(dataset: str, dest_path: str):
    """
    Downloads the specified Kaggle dataset using kagglehub and extracts or copies it to dest_path.
    
    Parameters:
        dataset (str): The Kaggle dataset identifier (e.g., "shivamb/vehicle-claim-fraud-detection").
        dest_path (str): The directory where the dataset should be placed.
    """
    downloaded_path = kagglehub.dataset_download(dataset)
    print(f"Dataset downloaded to: {downloaded_path}")
    
    os.makedirs(dest_path, exist_ok=True)
    
    if os.path.isfile(downloaded_path):
        # If it's a zip file, extract it.
        with zipfile.ZipFile(downloaded_path, 'r') as zip_ref:
            zip_ref.extractall(dest_path)
        print(f"Dataset extracted to: {dest_path}")
    elif os.path.isdir(downloaded_path):
        # If it's already a directory, copy its contents.
        print(f"Dataset is already extracted at: {downloaded_path}")
        for item in os.listdir(downloaded_path):
            source = os.path.join(downloaded_path, item)
            destination = os.path.join(dest_path, item)
            if os.path.isdir(source):
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(source, destination)
        print(f"Dataset copied to: {dest_path}")
    else:
        raise Exception("The downloaded path is neither a file nor a directory.")

if __name__ == "__main__":
    dataset_id = "shivamb/vehicle-claim-fraud-detection"
    dest_folder = os.path.join("..", "data")
    download_and_extract_dataset(dataset_id, dest_folder)
