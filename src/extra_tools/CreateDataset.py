import json
import os
import cv2
import parseAnnotationUtils
import random


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


def crop_frame(frame, top_left_corner, bottom_right_corner, frame_width, frame_height):
    """
    Crops the given frame based on the top-left and bottom-right corners.

    Parameters:
    frame (numpy.ndarray): The input frame to crop.
    top_left_corner (tuple): The top-left corner (x_min, y_min) of the bounding box.
    bottom_right_corner (tuple): The bottom-right corner (x_max, y_max) of the bounding box.

    Returns:
    numpy.ndarray: The cropped frame.
    """
    padding = 10
    x_min, y_min = top_left_corner
    x_max, y_max = bottom_right_corner

    x_min_pad = max(0, x_min - padding)
    y_min_pad = max(0, y_min - padding)
    x_max_pad = min(frame_width, x_max + padding)
    y_max_pad = min(frame_height, y_max + padding)

    max_dif = max(x_max_pad - x_min_pad, y_max_pad - y_min_pad)

    x_max_pad = min(frame_width, x_min_pad + max_dif)
    y_max_pad = min(frame_height, y_min_pad + max_dif)

    return frame[int(y_min_pad):int(y_max_pad), int(x_min_pad):int(x_max_pad)]


def resize_frame(frame, target_size=(224, 224), interpolation=cv2.INTER_LINEAR):
    return cv2.resize(frame, target_size, interpolation=interpolation)


def extract_and_annotate_frames(video_file, annotations, index_gesture_phase_preparation, index_gesture_phase_stroke,
                                index_gesture_phase_hold, index_gesture_phase_recovery):
    [gu, gp, bb] = annotations
    gesturePhases = asociateBBtoGPhase(bb, gp)

    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pathPreparation = 'dataset/preparation/'
    pathStroke = 'dataset/stroke/'
    pathHold = 'dataset/hold/'
    pathRecovery = 'dataset/recovery/'

    # list of all the frames
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

    for annotation in gesturePhases:
        start_time = gesturePhases[annotation]['time'][0]
        end_time = gesturePhases[annotation]['time'][1]
        start_frame = int(fps * timestamp_to_seconds(start_time))
        end_frame = int(fps * timestamp_to_seconds(end_time))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        clip_frames = []
        # if True:
        try:
            for frameIndex in range(start_frame, end_frame + 1):
                frame = frames[frameIndex]['frame']
                bbox = gesturePhases[annotation]['boundingBox']
                bbox = [int(bbox['position'][1]), int(bbox['position'][2]), int(bbox['position'][3]),
                        int(bbox['position'][4])]
                top_left_corner, bottom_right_corner = computeBBCoord(bbox)
                croppedFrame = crop_frame(frame, top_left_corner, bottom_right_corner, frame_width, frame_height)
                resizedFrame = resize_frame(croppedFrame)
                clip_frames.append(resizedFrame)

                # cv2.rectangle(frame, top_left_corner, bottom_right_corner, (0, 255, 0), 2)
                #
                # label = f"person {gesturePhases[annotation]['boundingBox']['person_id']}: {gesturePhases[annotation]['gestureType']}"
                # cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Write the frame to the video
                # frames[frameIndex]['frame'] = frame

            # Define the codec and create VideoWriter object

            clippedFrameWidth = clip_frames[0].shape[1]
            clippedFrameHeight = clip_frames[0].shape[0]

            output_video = ''
            if gesturePhases[annotation]['gestureType'] == 'preparation':
                output_video = pathPreparation + f"{index_gesture_phase_preparation}.mp4"
                index_gesture_phase_preparation += 1
            elif gesturePhases[annotation]['gestureType'] == 'stroke':
                output_video = pathStroke + f"{index_gesture_phase_stroke}.mp4"
                index_gesture_phase_stroke += 1
            elif gesturePhases[annotation]['gestureType'] == 'hold':
                output_video = pathHold + f"{index_gesture_phase_hold}.mp4"
                index_gesture_phase_hold += 1
            elif gesturePhases[annotation]['gestureType'] == 'recovery':
                output_video = pathRecovery + f"{index_gesture_phase_recovery}.mp4"
                index_gesture_phase_recovery += 1

            # clip_frames = adjust_clip_to_32_frames(clip_frames)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'MJPG', etc.
            out = cv2.VideoWriter(output_video, fourcc, fps, (clippedFrameWidth, clippedFrameHeight))

            if not out.isOpened():
                print(f"Error: Could not open the output video file: {output_video}")
                return

            for frame in clip_frames:
                out.write(frame)
            out.release()

        #     # print(f"Saved video: {output_video}")
        except Exception as e:
            print("clip failed")

        # if index_gesture_phase == 0:
        #     break

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    return index_gesture_phase_preparation, index_gesture_phase_stroke, index_gesture_phase_hold, index_gesture_phase_recovery


# creates annotation for unknown segments of the video whcih have length between 5-10 seconds. The annotation is saved in the dataset/unknown folder
# the time segment of an unknown segment is not annotated. It should not overlap with any other annotated segment
def annotate_unknown_segments(video_file, annotations, index_gesture_phase_unknown):
    [gu, gp, bb] = annotations
    gesturePhases = asociateBBtoGPhase(bb, gp)

    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pathUnknown = 'dataset/unknown/'
    unknown_segments = []

    for index_ann in range(len(gesturePhases) - 1):

        gesture_1_end = gesturePhases[index_ann]['time'][1]
        gesture_2_start = gesturePhases[index_ann + 1]['time'][0]
        current_segment = []
        bounding_box_1 = gesturePhases[index_ann]['boundingBox']['position']
        bounding_box_2 = gesturePhases[index_ann + 1]['boundingBox']['position']

        if gesture_2_start - gesture_1_end > 5:
            bounding_box = [int(bounding_box_1[1]), int(bounding_box_1[2]), int(bounding_box_1[3]),
                            int(bounding_box_1[4])]

            random_clip_length = int(random.randint(5, min(int(gesture_2_start - gesture_1_end), 10)) * fps)
            index_start = int(gesture_1_end * fps)
            index_end = index_start + random_clip_length

            for i in range(index_start, index_end):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break
                top_left_corner, bottom_right_corner = computeBBCoord(bounding_box)
                croppedFrame = crop_frame(frame, top_left_corner, bottom_right_corner, frame_width, frame_height)
                current_segment.append(croppedFrame)
            unknown_segments.append(current_segment)

        if gesture_2_start - gesture_1_end > 7:
            current_segment = []
            bounding_box = [int(bounding_box_2[1]), int(bounding_box_2[2]), int(bounding_box_2[3]),
                            int(bounding_box_2[4])]

            random_clip_length = int(random.randint(5, min(int(gesture_2_start - gesture_1_end - 2), 10)) * fps)
            index_end = int((gesture_2_start - 1) * fps)
            index_start = index_end - random_clip_length

            for i in range(index_start, index_end):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break
                top_left_corner, bottom_right_corner = computeBBCoord(bounding_box)
                croppedFrame = crop_frame(frame, top_left_corner, bottom_right_corner, frame_width, frame_height)
                current_segment.append(croppedFrame)
            unknown_segments.append(current_segment)

    ### save the unknown segments in the dataset/unknown folder ###
    for segment in unknown_segments:
        if segment:
            clip_frames = segment
            clippedFrameWidth = clip_frames[0].shape[1]
            clippedFrameHeight = clip_frames[0].shape[0]

            output_video = pathUnknown + f"{index_gesture_phase_unknown}.mp4"
            index_gesture_phase_unknown += 1

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (clippedFrameWidth, clippedFrameHeight))

            if not out.isOpened():
                print(f"Error: Could not open the output video file: {output_video}")
                return

            for frame in clip_frames:
                out.write(frame)
            out.release()

    cap.release()
    return index_gesture_phase_unknown


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


def timestamp_to_seconds(timestamp):
    return timestamp


def readAnnotationAllPersons(annotationPaths):
    annotationsPerPerson = []
    for annotationPath in annotationPaths:
        annotationPerson = parseAnnotationUtils.readAnnotationFile(annotationPath)
        annotationsPerPerson.append(annotationPerson)
    return annotationsPerPerson


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


def gesture_unit_extract_and_annotate(video_file, annotations, index_gesture_units, index_gesture_noting):
    [gu, gp, bb] = annotations
    gestureUnits = asociateBBtoGestureUnits(bb, gu)

    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pathGUnit = 'dataset/gunit/'
    pathNothing = 'dataset/nothing/'

    # Ensure output directories exist
    os.makedirs(pathGUnit, exist_ok=True)
    os.makedirs(pathNothing, exist_ok=True)

    # list of all the frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append({
            'frame': frame,
            'boundingBox': None
        })

    def extract_and_save_clip(aux_start_frame, aux_end_frame, output_path, clip_index, bb_crop):
        aux_clip_frames = []
        try:
            for aux_frameIndex in range(aux_start_frame, aux_end_frame + 1):
                aux_frame = frames[aux_frameIndex]['frame']
                aux_croppedFrame = crop_frame(aux_frame, (bb_crop[0], bb_crop[1]),
                                              (bb_crop[0] + bb_crop[2], bb_crop[1] + bb_crop[3]), frame_width,
                                              frame_height)
                aux_clip_frames.append(aux_croppedFrame)

            # Define the codec and create VideoWriter object
            a_clippedFrameWidth = aux_clip_frames[0].shape[1]
            a_clippedFrameHeight = aux_clip_frames[0].shape[0]

            output_video = output_path + f"{clip_index}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'MJPG', etc.
            out = cv2.VideoWriter(output_video, fourcc, fps, (a_clippedFrameWidth, a_clippedFrameHeight))

            if not out.isOpened():
                print(f"Error: Could not open the output video file: {output_video}")
                return

            for aux_frame in aux_clip_frames:
                out.write(aux_frame)
            out.release()

            print(f"Saved video: {output_video}")
        except Exception as e:
            print(f"Clip failed: {e}")

    def merge_bbs(bb1, bb2):
        aux_x = min(bb1[1], bb2[1])
        aux_y = min(bb1[2], bb2[2])
        aux_width = max(bb1[3], bb2[3])
        aux_height = max(bb1[4], bb2[4])
        return [aux_x, aux_y, aux_width, aux_height]

    last_end_frame = 0
    last_bound_box = None
    for annotation in gestureUnits:
        start_time = gestureUnits[annotation]['gestureUnit']['time'][0]
        end_time = gestureUnits[annotation]['gestureUnit']['time'][1]
        start_frame = int(fps * timestamp_to_seconds(start_time))
        end_frame = int(fps * timestamp_to_seconds(end_time))
        current_bound_box = gestureUnits[annotation]['boundingBox'][-1]['position']

        # Save the "nothing" clip if there's a gap before the current gesture unit
        if start_frame - last_end_frame > 2 * fps:
            if (last_bound_box == None):
                last_bound_box = current_bound_box

            merged_bb = merge_bbs(last_bound_box, current_bound_box)

            extract_and_save_clip(last_end_frame, start_frame - 1, pathNothing, index_gesture_noting, merged_bb)
            index_gesture_noting += 1
            last_bound_box = current_bound_box

        last_end_frame = end_frame + 1
        clip_frames = []
        try:
            for frameIndex in range(start_frame, end_frame + 1):
                frame = frames[frameIndex]['frame']
                bbox = None
                frameTime = int(frameIndex / fps)
                # set the bbox as the bounding box that has the time closest to the frame time
                minDiff = float('inf')
                for bb in gestureUnits[annotation]['boundingBox']:
                    if abs(bb['time'][0] - frameTime) < minDiff:
                        minDiff = abs(bb['time'][0] - frameTime)
                        bbox = bb

                bbox = [int(bbox['position'][1]), int(bbox['position'][2]), int(bbox['position'][3]),
                        int(bbox['position'][4])]
                top_left_corner, bottom_right_corner = computeBBCoord(bbox)
                croppedFrame = crop_frame(frame, top_left_corner, bottom_right_corner, frame_width, frame_height)
                resizedFrame = resize_frame(croppedFrame)
                clip_frames.append(resizedFrame)

            # Define the codec and create VideoWriter object
            clippedFrameWidth = clip_frames[0].shape[1]
            clippedFrameHeight = clip_frames[0].shape[0]

            output_video = pathGUnit + f"{index_gesture_units}.mp4"
            index_gesture_units += 1

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'MJPG', etc.
            out = cv2.VideoWriter(output_video, fourcc, fps, (clippedFrameWidth, clippedFrameHeight))

            if not out.isOpened():
                print(f"Error: Could not open the output video file: {output_video}")
                return

            for frame in clip_frames:
                out.write(frame)
            out.release()

            print(f"Saved video: {output_video}")
        except Exception as e:
            print(f"Clip failed: {e}")

    # Save the "nothing" clip if there's remaining frames after the last gesture unit
    if len(frames) - last_end_frame > 2 * fps:
        extract_and_save_clip(last_end_frame, len(frames) - 1, pathNothing, index_gesture_noting, last_bound_box)
        index_gesture_noting += 1

    # Release the VideoCapture object
    cap.release()
    return [index_gesture_units, index_gesture_noting]


# def gesture_unit_extract_and_annotate(video_file, annotations, index_gesture_units):
#     [gu, gp, bb] = annotations
#     gestureUnits = asociateBBtoGestureUnits(bb, gu)
#
#     cap = cv2.VideoCapture(video_file)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     pathGUnit = 'dataset/gunit/'
#     pathNothing = 'dataset/nothing/'
#
#     # list of all the frames
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append({
#             'frame': frame,
#             'boundingBox': None
#         })
#
#
#     for annotation in gestureUnits:
#         start_time = gestureUnits[annotation]['time'][0]
#         end_time = gestureUnits[annotation]['time'][1]
#         start_frame = int(fps * timestamp_to_seconds(start_time))
#         end_frame = int(fps * timestamp_to_seconds(end_time))
#
#         cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
#
#         clip_frames = []
#         try:
#             for frameIndex in range(start_frame, end_frame + 1):
#                 frame = frames[frameIndex]['frame']
#                 bbox = None
#                 frameTime = int(frameIndex / fps)
#                 # set the bbox as the bounding box that has the tine closest to the frame time
#                 minDiff = 10000000000
#                 for bb in gestureUnits[annotation]['boundingBox']:
#                     if abs(bb['time'][0] - frameTime) < minDiff:
#                         minDiff = abs(bb['time'][0] - frameTime)
#                         bbox = bb
#
#                 bbox = [int(bbox['position'][1]), int(bbox['position'][2]), int(bbox['position'][3]),
#                         int(bbox['position'][4])]
#                 top_left_corner, bottom_right_corner = computeBBCoord(bbox)
#                 croppedFrame = crop_frame(frame, top_left_corner, bottom_right_corner, frame_width, frame_height)
#                 resizedFrame = resize_frame(croppedFrame)
#                 clip_frames.append(resizedFrame)
#             # Define the codec and create VideoWriter object
#
#             clippedFrameWidth = clip_frames[0].shape[1]
#             clippedFrameHeight = clip_frames[0].shape[0]
#
#             output_video = pathGUnit + f"{index_gesture_units}.mp4"
#             index_gesture_units += 1
#
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'MJPG', etc.
#             out = cv2.VideoWriter(output_video, fourcc, fps, (clippedFrameWidth, clippedFrameHeight))
#
#             if not out.isOpened():
#                 print(f"Error: Could not open the output video file: {output_video}")
#                 return
#
#             for frame in clip_frames:
#                 out.write(frame)
#             out.release()
#
#         #     # print(f"Saved video: {output_video}")
#         except Exception as e:
#             print("clip failed")
#
#         # if index_gesture_phase == 0:
#         #     break
#
#     # Release the VideoCapture and VideoWriter objects
#     cap.release()
#     return index_gesture_units
#

if __name__ == "__main__":
    video_paths = [
        {
            'video_path': 'annotations/cam2/progressVid2-seg9/vid2-seg9-scaled-denoised.mp4',
            'annotation_paths': [
                'annotations/cam2/progressVid2-seg9/person1_annotated.json',
                'annotations/cam2/progressVid2-seg9/person2_annotated.json',
                'annotations/cam2/progressVid2-seg9/person3_annotated.json',
                'annotations/cam2/progressVid2-seg9/person4_annotated.json',
                'annotations/cam2/progressVid2-seg9/person5_annotated.json',
                'annotations/cam2/progressVid2-seg9/person6_annotated.json',
                'annotations/cam2/progressVid2-seg9/person7_annotated.json',
                'annotations/cam2/progressVid2-seg9/person8_annotated.json'
            ]
        },
        {
            'video_path': 'annotations/cam2/progressVid3-seg1/vid3-seg1-scaled-denoised.mp4',
            'annotation_paths': [
                'annotations/cam2/progressVid3-seg1/person1_annotated.json',
                'annotations/cam2/progressVid3-seg1/person2_annotated.json',
                'annotations/cam2/progressVid3-seg1/person3_annotated.json'
            ]
        },
        {
            'video_path': 'annotations/cam4/vid2-seg8-scaled-denoised.mp4',
            'annotation_paths': [
                'annotations/cam4/person1_annotated.json',
            ]
        },
        {
            'video_path': 'annotations/cam6/vid2-seg8-scaled-denoised.mp4',
            'annotation_paths': [
                'annotations/cam6/person1_annotated.json',
            ]
        },
    ]

    # video_path = 'annotations/cam2/progressVid2-seg9/vid2-seg9-scaled-denoised.mp4'
    # annotationPaths = [
    #     'annotations/cam2/progressVid2-seg9/person1_annotated.json',
    #     'annotations/cam2/progressVid2-seg9/person2_annotated.json',
    #     'annotations/cam2/progressVid2-seg9/person3_annotated.json',
    #     'annotations/cam2/progressVid2-seg9/person4_annotated.json',
    #     'annotations/cam2/progressVid2-seg9/person5_annotated.json',
    #     'annotations/cam2/progressVid2-seg9/person6_annotated.json',
    #     'annotations/cam2/progressVid2-seg9/person7_annotated.json',
    #     'annotations/cam2/progressVid2-seg9/person8_annotated.json'
    # ]

    index_gesture_phase_preparation = 0
    index_gesture_phase_stroke = 0
    index_gesture_phase_hold = 0
    index_gesture_phase_recovery = 0
    index_gesture_phase_unknown = 0
    index_gesture_units = 0
    index_gesture_noting = 0

    if True:
        for video_path in video_paths:
            video_file = video_path['video_path']
            annotationPaths = video_path['annotation_paths']
            # video_file = video_path
            annotations = readAnnotationAllPersons(annotationPaths)

            # if True:
            for i in range(len(annotations)):
                # [index_gesture_phase_preparation, index_gesture_phase_stroke, index_gesture_phase_hold,
                #  index_gesture_phase_recovery] = extract_and_annotate_frames(video_file, annotations[0],
                #                                                              index_gesture_phase_preparation,
                #                                                              index_gesture_phase_stroke,
                #                                                              index_gesture_phase_hold,
                #                                                              index_gesture_phase_recovery)
                # index_gesture_phase_unknown = annotate_unknown_segments(video_file, annotations[0],
                #                                                         index_gesture_phase_unknown)
                [index_gesture_units, index_gesture_noting] = gesture_unit_extract_and_annotate(video_file,
                                                                                                annotations[i],
                                                                                                index_gesture_units,
                                                                                                index_gesture_noting)
                print("!!!!!!!!!!!!!Annotated person " + str(i + 1))

            print("----Annotated video " + video_file)
