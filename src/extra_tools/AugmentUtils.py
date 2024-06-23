import copy

import cv2
import numpy as np
import random
import os
from PIL import Image, ImageEnhance


# Define augmentation functions
def random_crop(frames, frame_size=(224, 224)):
    height, width, _ = frames[0].shape
    crop_len = random.randint(0, 32)
    for i in range(len(frames)):
        frame = frames[i]
        cropped_frame = frame[crop_len:height - crop_len, crop_len:width - crop_len]
        frames[i] = cv2.resize(cropped_frame, frame_size)
    return frames


def flip_image(frames):
    for i in range(len(frames)):
        frames[i] = cv2.flip(frames[i], 1)
    return frames


def color_jitter(frames):
    for i in range(len(frames)):
        image = Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))
        frames[i] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return frames


def gaussian_noise(frames):
    for i in range(len(frames)):
        frame = frames[i]
        row, col, ch = frame.shape
        mean = 0
        var = 0.01
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy_frame = frame + gauss.reshape(row, col, ch)
        noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
        frames[i] = noisy_frame
    return frames


def gaussian_blur(frames, ksize=(5, 5)):
    for i in range(len(frames)):
        frames[i] = cv2.GaussianBlur(frames[i], ksize, 0)
    return frames


def rotate(frames, angle_range=(-90, 90)):
    angle = random.uniform(*angle_range)
    height, width = frames[0].shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    for i in range(len(frames)):
        frames[i] = cv2.warpAffine(frames[i], matrix, (width, height))
    return frames


# Define a function to apply a random subset of augmentations
def apply_random_augmentations(frames):
    augmentations = [random_crop, flip_image, color_jitter, gaussian_noise, gaussian_blur, rotate]
    selected_augmentations = random.sample(augmentations, random.randint(1, len(augmentations)))
    frames = selected_augmentations(frames)
    return frames


# Function to process and save augmented videos
def augment_and_save(video_path, output_path, num_augmentations=5, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Original video saving
    save_video(frames, output_path, "original", video_path, target_size)

    # Single augmentation videos
    index_aug = 0
    for aug in [random_crop, flip_image, color_jitter, gaussian_noise, gaussian_blur, rotate]:
        single_augmented_frames = copy.deepcopy(frames)
        single_augmented_frames = aug(single_augmented_frames)
        save_video(single_augmented_frames, output_path, f"aug_{index_aug}", video_path, target_size)
        index_aug += 1

    # Multiple augmentations videos
    subsetLength = random.randint(2, num_augmentations)
    subsetAugmentations = random.sample([random_crop, flip_image, color_jitter, gaussian_noise, gaussian_blur, rotate], 5)
    multiple_augmented_frames = copy.deepcopy(frames)
    for aug in subsetAugmentations:
        multiple_augmented_frames = aug(multiple_augmented_frames)

    save_video(multiple_augmented_frames, output_path, f"multiple_aug", video_path, target_size)


def save_video(frames, output_path, augmentation_type, video_path, target_size):
    frames_array = np.array(frames)
    output_video_path = os.path.join(output_path, f"{augmentation_type}_{os.path.basename(video_path)}")
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, target_size)
    for frame in frames_array:
        out.write(frame)
    out.release()


if __name__ == '__main__':
    input_video = 'datasetOld/stroke/10.mp4'
    output_dir = 'old_data/augmented_videos'
    os.makedirs(output_dir, exist_ok=True)
    augment_and_save(input_video, output_dir)
