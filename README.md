# Hand gestures classification in crowded environments
 
## Abstract
Hand gestures play a crucial role in communication, especially in social interactions. This research investigates the viability of using coding schemes to describe hand gestures and how accurately they can be classified in crowded environments by using f ine-tuned visual transformers such as VideoMAE. The dataset used during training is based on the Conflab dataset and contains top-view video recordings of social interactions in a crowded social setting. The videos are manually annotated for gesture phases (preparation, hold, stroke, recovery) and gesture units. The two classifiers obtain high accuracies after fine tuning, with an overall accuracy of 95% for the gesture phase classification and 93% for classifying whether a clip is a gesture unit or not. These f indings suggest that the proposed approach is effective in crowded environments and can be adapted for real-time applications.


## How to run the code

The fine-tuning process happens in two notebooks. Those are `train_gesture_phase.ipynb` and `train_gesture_unit.ipynb`. The first notebook is used to fine-tune the model for gesture phase classification, and the second notebook is used to fine-tune the model for gesture unit classification. 


## Dataset
The dataset used in this reaseach is not provided, but the folders `dataset_gphase` and `dataset_gunit` contain the structure of the dataset used in this research. The dataset is based on the Conflab dataset and contains top-view video recordings of social interactions in a crowded social setting. The videos are manually annotated for gesture phases (preparation, hold, stroke, recovery) and gesture units

## Prepare the dataset

The dataset is prepared in the `create_dataset.ipynb notebool`. This notebook is used to create the dataset for the gesture phase and gesture unit classification. The dataset is created by extracting the frames from the videos and saving them in the appropriate folders. The dataset is then split into training and validation sets.

## Extra tools
The `extra_tools` files contains code used to build the `create_dataset.ipynb` file, and some extra function for creating annotated videos, fake datasets and so on.
The only important file in this folder is the `CreateDataset.py` file, which contains all the code used to create the dataset for the gesture phase and gesture unit classification.