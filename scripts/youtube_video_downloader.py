import json
from pytubefix import YouTube
import os

# Load the train data
with open('../data/MSASL_test.json', 'r') as f:
    train_data = json.load(f)

# Directory to save the downloaded videos
download_dir = 'test_videos'

# Directory to save failed downloads
failed_downloads_file = 'failed_downloads.json'

# Create the main download directory if it doesn't exist
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Initialize list to store metadata of failed downloads
failed_downloads = []


# Function to download video
def download_video(url, output_path, filename):
    try:
        yt = YouTube(url)
        stream = yt.streams.get_highest_resolution()
        stream.download(output_path, filename=filename)
        print(f"Downloaded: {url} as {filename}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


# Iterate through the train data, download videos, and save metadata
for item in train_data:
    url = item.get('url')
    if url:
        # Create a directory for the ASL sign if it doesn't exist
        sign_dir = os.path.join(download_dir, item['clean_text'])
        if not os.path.exists(sign_dir):
            os.makedirs(sign_dir)

        # Generate a base file name
        base_file_name = f"{item['file']}_{item['signer_id']}.mp4"
        file_name = base_file_name
        file_path = os.path.join(sign_dir, file_name)
        counter = 1

        # Check if the file name already exists, if so, append a counter
        while os.path.exists(file_path):
            file_name = f"{item['file']}_{item['signer_id']}_{counter}.mp4"
            file_path = os.path.join(sign_dir, file_name)
            counter += 1

        # Attempt to download the video
        if download_video(url, sign_dir, file_name):
            # Save metadata for the video
            metadata = {
                "file": file_path,
                "org_text": item["org_text"],
                "clean_text": item["clean_text"],
                "start_time": item["start_time"],
                "end_time": item["end_time"],
                "signer_id": item["signer_id"],
                "signer": item["signer"],
                "start_frame": item["start"],
                "end_frame": item["end"],
                "label": item["label"],
                "height": item["height"],
                "width": item["width"],
                "fps": item["fps"],
                "box": item["box"],
                "url": item["url"]
            }

            # Save the metadata to a JSON file within the same directory as the video
            metadata_file = os.path.join(sign_dir, f"{os.path.splitext(file_name)[0]}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
        else:
            # If the download fails, store the item in failed_downloads
            failed_downloads.append(item)

# Save the failed downloads metadata to a JSON file
with open(failed_downloads_file, 'w') as f:
    json.dump(failed_downloads, f, indent=4)

print(f"All videos and metadata have been processed.")
print(f"Failed downloads have been saved to {failed_downloads_file}")
