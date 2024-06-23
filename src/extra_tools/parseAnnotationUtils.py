import numpy as np
import json

# The function takes the metadata and attributes as input and returns a list of dictionaries
# containing the gesture units, gesture phases, and bounding boxes.
    # id = 1 => PersonLocation, options = 0  (person1), 1, 2, 3 for each person
    # id = 2 => GestureUnit, options = "0": "g-unit"
    # id = 3 => GesturePhase, "options": {
    #             "0": "preparation",
    #             "1": "stroke",
    #             "2": "recovery",
    #             "3": "hold",
    #             "4": "incompleteStroke"
    #         },

    # Gesture Unit
    # 'id': 0,
    # 'timeStart': 0,
    # 'timeEnd': 25,

    # Gesture Phase
    # 'id': 0,
    # 'timeStart': 0,
    # 'timeEnd': 25,
    # 'gestureType': "recovery",

    # Bounding Box
    # 'id': 0,
    # 'position': [0.0, 0.0, 0.0, 0.0],
    # 'time': 0,
    # 'gesturePhase': "recovery",
def processMetadata(metadata, attributes):
    gestureUnits = {}
    gesturePhases = {}
    boundingBoxes = {}
    idGU = 0
    idGP = 0
    idBB = 0

    for annot in metadata:
        # it can be either the time interval of the gesture, either the time at which the bounding box appears
        # (you can deduce the frame number from this)
        zValue = metadata[annot]['z']

        # the bounding box coordinates. If it is temporal segment is empty
        xyValue = metadata[annot]['xy']

        # defines the value of each attribute for this (z, xy) combination
        # "1":"1"       # the value for attribute-id="1" is one of its option with id "1" (i.e. Activity = Break Egg)
        avValue = metadata[annot]['av']
        phasesList = []
        if len(avValue) == 1:
            attrKey = list(avValue.keys())[0]
            if(len(zValue)!=2):
                continue
            if(attrKey == '2'):
                gestureUnits[idGU] = {
                        'time': zValue,
                }
                idGU += 1
                # print("Gesture Unit annotation: ", metadata[annot])
            else:
                gesturePhases[idGP] = {
                        'time': zValue,
                        'gestureType': attributes['3']['options'][avValue[attrKey]]
                    }
                idGP += 1
                # print("Gesture Phase annotation: ", metadata[annot])
        else:
            boundingBoxes[idBB] = {
                    'position': xyValue,
                    'time': zValue,
                    'gesturePhase': attributes['3']['options'][avValue['3']],
                    'person_id': avValue['1']
                }
            idBB += 1
            # print("bounding box annotation: ", metadata[annot])
        # print("\n\n-------------------\n\n")

    # print("Gesture Units: \n")
    # for gu in gestureUnits:
    #     print(gestureUnits[gu])
    # print("\n\n-------------------\n\n")
    #
    # print("\n\n\nGesture Phases: \n")
    # for gp in gesturePhases:
    #     print(gesturePhases[gp])
    # print("\n\n-------------------\n\n")
    #
    # print("\n\n\nBounding Boxes: \n")
    # for bb in boundingBoxes:
    #     print(boundingBoxes[bb])

    return [gestureUnits, gesturePhases, boundingBoxes]


# The function reads the annotation file and processes the metadata
def readAnnotationFile(annotationFilePath):
    try:
        with open(annotationFilePath, 'r') as file:
            data = json.load(file)
        attributes = data['attribute']
        # pretty_json = json.dumps(attributes, indent=4)
        # print(pretty_json)

        [gestureUnits, gesturePhases, boundingBoxes] = processMetadata(data['metadata'], data['attribute'])
        return [gestureUnits, gesturePhases, boundingBoxes]

    except FileNotFoundError:
        print("The file was not found")
    except json.JSONDecodeError:
        print("The file contains invalid JSON")




if __name__ == "__main__":
    readAnnotationFile('annotations/cam2/progressVid2-seg9/person1_annotated.json')

