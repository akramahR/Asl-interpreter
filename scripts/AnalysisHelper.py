from collections import Counter

from src.preprocessing import load_video_paths_and_labels
from videoGenerator import *

# Load video paths, labels, start_times, and end_times
data_dir = '../scripts/videos'
video_paths, labels, start_times, end_times, fps_list = load_video_paths_and_labels(data_dir)

# Count the occurrences of each label
label_counts = Counter(labels)

# Sort the labels by the number of occurrences in descending order
sorted_label_counts = label_counts.most_common()

# Display the labels with their respective counts
print("Labels and their counts (most to least):")
for label, count in sorted_label_counts:
    print(f"{label}: {count}")

# Find the label(s) with the maximum number of videos
max_count = max(label_counts.values())
most_common_labels = [label for label, count in label_counts.items() if count == max_count]

print(f"\nLabel(s) with the most videos ({max_count} occurrences): {', '.join(most_common_labels)}")
