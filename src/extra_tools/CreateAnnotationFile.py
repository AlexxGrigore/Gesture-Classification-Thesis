import os
import csv

if __name__ == '__main__':
    # Define the dataset path and gesture labels
    dataset_path = 'old_data/dataset'
    split_folders = ['train', 'val', 'test']
    gesture_folders = ['preparation', 'stroke', 'hold', 'recovery']
    labels = {'preparation': 0, 'stroke': 1, 'hold': 2, 'recovery': 3}

    # Collect video paths and labels
    for split_folder in split_folders:
        annotations = []
        for gesture in gesture_folders:
            folder_path = os.path.join(dataset_path, split_folder, gesture)
            for video_file in os.listdir(folder_path):
                if video_file.endswith('.mp4'):  # Adjust the extension if needed
                    video_path = os.path.join(folder_path, video_file)
                    annotations.append({"video": video_path, "label": labels[gesture]})

        # Save annotations to CSV
        csv_file_path = os.path.join(dataset_path, split_folder, 'annotations.csv')
        with open(csv_file_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['video', 'label'])
            writer.writeheader()
            writer.writerows(annotations)














# import os
# import json
# import csv
#
#
# if __name__ == '__main__':
#     # Define the dataset path and gesture labels
#     dataset_path = 'dataset'
#     split_folders = ['train', 'val', 'test']
#     gesture_folders = ['preparation', 'stroke', 'hold', 'recovery']
#     labels = {'preparation': 0, 'stroke': 1, 'hold': 2, 'recovery': 3}
#
#     # Collect video paths and labels
#     for split_folder in split_folders:
#         annotations = []
#         for gesture in gesture_folders:
#             folder_path = os.path.join(dataset_path, split_folder, gesture)
#             for video_file in os.listdir(folder_path):
#                 if video_file.endswith('.mp4'):  # Adjust the extension if needed
#                     video_path = os.path.join(folder_path, video_file)
#                     annotations.append({"video": video_path, "label": gesture})
#
#         # Save annotations to JSON
#         with open(os.path.join(dataset_path, split_folder, 'annotations.json'), 'w') as json_file:
#             json.dump(annotations, json_file, indent=4)
