import cv2
import os

def count_frames(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return None

    # Get the total number of frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Release the video capture object
    video.release()

    return total_frames

def average_frames_in_folder(folder_path):
    total_frames = 0
    video_count = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add other video formats if needed
            video_path = os.path.join(folder_path, filename)
            frame_count = count_frames(video_path)
            if frame_count is not None:
                total_frames += frame_count
                video_count += 1

    if video_count == 0:
        return 0

    average_frames = total_frames / video_count
    return average_frames

if __name__ == '__main__':
    folder_path = 'old_data/datasetOld/recovery'
    average_frame_count = average_frames_in_folder(folder_path)
    print(f'Average number of frames per video: {average_frame_count}')

