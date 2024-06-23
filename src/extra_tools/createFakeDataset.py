import os
from moviepy.video.io.VideoFileClip import VideoFileClip


def split_video_into_clips(video_path, output_folder, clip_duration=1):
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the video
    video = VideoFileClip(video_path)
    video_duration = int(video.duration)

    # Split the video into clips
    clip_count = 0
    for start_time in range(0, video_duration, clip_duration):
        end_time = min(start_time + clip_duration, video_duration)
        clip = video.subclip(start_time, end_time)
        clip_filename = os.path.join(output_folder, f'{clip_count}.mp4')
        clip.write_videofile(clip_filename, codec="libx264")
        clip_count += 1


if __name__ == "__main__":
    video_path = "dumb_video.mp4"  # Replace with the path to your video file
    output_folder = "dataset/hold"  # The single output folder for all clips

    split_video_into_clips(video_path, output_folder)
