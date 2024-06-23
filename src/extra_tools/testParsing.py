import json
import cv2
def load_annotations(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data


def parse_annotations(data):
    parsed_data = {}
    for video in data:
        video_id = video['video_id']
        parsed_data[video_id] = []
        for annotation in video['annotations']:
            parsed_annotation = {
                'timestamp': annotation['timestamp'],
                'person_id': annotation['person_id'],
                'bounding_box': annotation['bounding_box'],
                'gesture_phase': annotation['gesture_phase'],
                'time_interval': annotation['time_interval']
            }
            parsed_data[video_id].append(parsed_annotation)
    return parsed_data


# Function to extract frames and annotate bounding boxes
def extract_and_annotate_frames(video_file, annotations, output_dir):
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)

    for annotation in annotations:
        start_time = annotation['time_interval'][0]
        end_time = annotation['time_interval'][1]
        start_frame = int(fps * timestamp_to_seconds(start_time))
        end_frame = int(fps * timestamp_to_seconds(end_time))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_num in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break

            bbox = annotation['bounding_box']
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            label = f"{annotation['person_id']}: {annotation['gesture_phase']}"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            output_file = os.path.join(output_dir, f"{frame_num:06d}.jpg")
            cv2.imwrite(output_file, frame)

    cap.release()


def timestamp_to_seconds(timestamp):
    h, m, s = map(int, timestamp.split(':'))
    return h * 3600 + m * 60 + s


# Example usage
annotations_file = 'annotations.json'
video_path = 'path_to_videos'
output_dir = 'old_data/annotated_frames'

annotations_data = load_annotations(annotations_file)
parsed_annotations = parse_annotations(annotations_data)

# Process each video
for video_id, annotations in parsed_annotations.items():
    video_file = os.path.join(video_path, f"{video_id}.mp4")
    extract_and_annotate_frames(video_file, annotations, output_dir)
