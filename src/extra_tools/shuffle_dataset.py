import os
import random
import shutil
from collections import defaultdict


def compute_label_distribution(base_path):
    splits = ["train", "val", "test"]
    subfolders = ["hold", "recovery", "preparation", "stroke", "unknown"]

    # Dictionary to hold the count of each label in each split
    distribution = {split: defaultdict(int) for split in splits}
    total_counts = {split: 0 for split in splits}

    # Iterate over each split
    for split in splits:
        for subfolder in subfolders:
            folder_path = os.path.join(base_path, split, subfolder)

            # Verify subfolder path exists
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"The subfolder path {folder_path} does not exist.")

            # Count the number of files in the subfolder
            num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            distribution[split][subfolder] += num_files
            total_counts[split] += num_files

    # Print the distribution
    for split in splits:
        print(f"Distribution for {split}:")
        for subfolder in subfolders:
            count = distribution[split][subfolder]
            total = total_counts[split]
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {subfolder}: {count} files ({percentage:.2f}%)")
        print()

# Function to split the data
def split_data(files, split_ratio):
    random.shuffle(files)
    total_files = len(files)
    train_split = int(total_files * split_ratio[0])
    val_split = int(total_files * split_ratio[1])
    train_files = files[:train_split]
    val_files = files[train_split:train_split + val_split]
    test_files = files[train_split + val_split:]
    return train_files, val_files, test_files

if __name__ == '__main__':
    # Paths
    # dataset_path = "dataset"
    # output_path = "dataset_final"
    # subfolders = ["hold", "recovery", "preparation", "stroke", "unknown"]
    #
    # # Create the final dataset structure
    # for split in ["train", "val", "test"]:
    #     for subfolder in subfolders:
    #         os.makedirs(os.path.join(output_path, split, subfolder), exist_ok=True)
    #
    # # Process each subfolder
    # for subfolder in subfolders:
    #     folder_path = os.path.join(dataset_path, subfolder)
    #     files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    #
    #     # Split files
    #     train_files, val_files, test_files = split_data(files, (0.7, 0.15, 0.15))
    #
    #     # Copy files to the corresponding directories
    #     for f in train_files:
    #         shutil.copy(os.path.join(folder_path, f), os.path.join(output_path, "train", subfolder, f))
    #     for f in val_files:
    #         shutil.copy(os.path.join(folder_path, f), os.path.join(output_path, "val", subfolder, f))
    #     for f in test_files:
    #         shutil.copy(os.path.join(folder_path, f), os.path.join(output_path, "test", subfolder, f))
    #
    # print("Dataset split completed!")

    # Assuming 'src/dataset_final' is the base path
    base_path = "old_data/dataset_final"
    compute_label_distribution(base_path)







