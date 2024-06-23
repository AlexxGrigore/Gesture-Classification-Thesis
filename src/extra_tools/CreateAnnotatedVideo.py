import json
import os
import cv2
import parseAnnotationUtils
import random
import cv2

def asociateBBtoGPhase(boundingBoxes, gesturePhases):
    newGp = {}
    indexGp = 0
    for bb in boundingBoxes:

        for gp in gesturePhases:
            if abs(boundingBoxes[bb]['time'][0] - gesturePhases[gp]['time'][0]) <= 0.5:
                if (boundingBoxes[bb]['position'][3] < 10) or (boundingBoxes[bb]['position'][4] < 10):
                    continue
                else:
                    newGp[indexGp] = gesturePhases[gp]
                    newGp[indexGp]['boundingBox'] = boundingBoxes[bb]
                    indexGp += 1
    return newGp
def asociateBBtoGestureUnits(boundingBoxes, gestureUnits):
    newGu = {}

    for bb in boundingBoxes:
        indexGu = 0
        for gu in gestureUnits:
            if ((abs(boundingBoxes[bb]['time'][0] - gestureUnits[gu]['time'][0]) <= 0.5)
                    or (abs(boundingBoxes[bb]['time'][0] - gestureUnits[gu]['time'][1]))
                    or (gestureUnits[gu]['time'][0] <= boundingBoxes[bb]['time'][0] <= gestureUnits[gu]['time'][1])):

                if (boundingBoxes[bb]['position'][3] < 10) or (boundingBoxes[bb]['position'][4] < 10):
                    continue
                else:
                    if indexGu not in newGu:
                        newGu[indexGu] = {
                            'gestureUnit': gestureUnits[gu],
                            'boundingBox': []
                        }
                    newGu[indexGu]['boundingBox'].append(boundingBoxes[bb])
                    indexGu += 1

    # make all bounding boxes of a gesture unit to be the same length. the new width and the new height would be the
    # maximum width and height of all bounding boxes
    for gu in newGu:
        max_width = 0
        max_height = 0
        for bb in newGu[gu]['boundingBox']:

            width = bb['position'][3]
            height = bb['position'][4]

            if width > max_width:
                max_width = width
            if height > max_height:
                max_height = height
        for bb in newGu[gu]['boundingBox']:
            bb['position'][3] = max_width
            bb['position'][4] = max_height

    return newGu


def computeBBCoord(bb):
    x_top_left = bb[0]
    y_top_left = bb[1]
    width = bb[2]
    height = bb[3]

    # Calculate bottom-right coordinates
    x_bottom_right = x_top_left + width
    y_bottom_right = y_top_left + height

    # Corner coordinates
    top_left_corner = (x_top_left, y_top_left)
    bottom_right_corner = (x_bottom_right, y_bottom_right)
    return top_left_corner, bottom_right_corner

def extract_and_annotate_frames(video_file, annotations1, annotations2):
    [gu1, gp1, bb1] = annotations1
    [gu2, gp2, bb2] = annotations2

    gesturePhases1 = asociateBBtoGPhase(bb1, gp1)
    gesturePhases2 = asociateBBtoGPhase(bb2, gp2)

    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output path for annotated video
    output_video_file = 'annotated_video1.mp4'

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    # List of all the frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append({
            'frame': frame,
            'boundingBox': None
        })

    index_gesture_phase = 0

    for annotation in gesturePhases1:
        start_time = gesturePhases1[annotation]['time'][0]
        end_time = gesturePhases1[annotation]['time'][1]
        start_frame = int(fps * start_time)
        end_frame = int(fps * end_time)

        for frameIndex in range(start_frame, end_frame + 1):
            frame = frames[frameIndex]['frame']
            bbox = gesturePhases1[annotation]['boundingBox']
            bbox = [int(bbox['position'][1]), int(bbox['position'][2]), int(bbox['position'][3]),
                    int(bbox['position'][4])]
            top_left_corner, bottom_right_corner = computeBBCoord(bbox)

            # Draw bounding box on the frame
            cv2.rectangle(frame, top_left_corner, bottom_right_corner, (0, 255, 0), 2)

            # Create label for the bounding box
            label = f"Person {gesturePhases1[annotation]['boundingBox']['person_id']}: {gesturePhases1[annotation]['gestureType']}"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Write the annotated frame to the video
            frames[frameIndex]['frame'] = frame

    for annotation in gesturePhases2:
        start_time = gesturePhases2[annotation]['time'][0]
        end_time = gesturePhases2[annotation]['time'][1]
        start_frame = int(fps * start_time)
        end_frame = int(fps * end_time)

        for frameIndex in range(start_frame, end_frame + 1):
            frame = frames[frameIndex]['frame']
            bbox = gesturePhases2[annotation]['boundingBox']
            bbox = [int(bbox['position'][1]), int(bbox['position'][2]), int(bbox['position'][3]),
                    int(bbox['position'][4])]
            top_left_corner, bottom_right_corner = computeBBCoord(bbox)

            # Draw bounding box on the frame
            cv2.rectangle(frame, top_left_corner, bottom_right_corner, (0, 0, 255), 2)

            # Create label for the bounding box
            label = f"Person {gesturePhases2[annotation]['boundingBox']['person_id']}: gunit"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Write the annotated frame to the video
            frames[frameIndex]['frame'] = frame

    for frameIndex in range(len(frames)):
        out.write(frames[frameIndex]['frame'])

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()



if __name__ == '__main__':
    # Define the dataset path and gesture labels
    video_file = 'annotations/cam2/progressVid2-seg9/vid2-seg9-scaled-denoised.mp4'
    annotation_file1 = 'annotations/cam2/progressVid2-seg9/person1_annotated.json'
    annotation_file2 = 'annotations/cam2/progressVid2-seg9/person6_annotated.json'
    annotations1 = parseAnnotationUtils.readAnnotationFile(annotation_file1)
    annotations2 = parseAnnotationUtils.readAnnotationFile(annotation_file2)
    extract_and_annotate_frames(video_file, annotations1, annotations2)