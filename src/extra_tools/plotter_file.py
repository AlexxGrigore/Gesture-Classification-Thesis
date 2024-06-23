import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Data for Classifier 1
    epochs_1 = [0, 1, 2, 3, 4]
    training_loss_1 = [0.6685, 0.5348, 0.3395, 0.3607, 0.2436]
    validation_loss_1 = [0.6297, 0.4292, 0.4852, 0.4507, 0.4433]

    # Data for Classifier 2
    epochs_2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    training_loss_2 = [0.8549, 0.6437, 0.6226, 0.5883, 0.5136, 0.4284, 0.3474, 0.2859, 0.3059, 0.2582]
    validation_loss_2 = [0.6774, 0.7254, 0.6527, 0.6389, 0.5330, 0.4506, 0.3974, 0.3822, 0.3763, 0.3738]

    # Data for Classifier 3
    epochs_3 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    training_loss_3 = [0.6625, 0.6423, 0.6252, 0.5455, 0.4912, 0.383, 0.2696, 0.1547, 0.3174, 0.1324, 0.0265, 0.2346, 0.0597, 0.0103, 0.0358, 0.0047, 0.0054, 0.0024, 0.0027]
    validation_loss_3 = [0.6896, 0.6737, 0.6100, 0.5558, 0.4599, 0.3426, 0.3441, 0.3526, 0.3234, 0.5808, 0.6995, 0.7486, 0.7536, 0.4090, 0.8257, 0.6291, 0.8420, 0.8389, 0.7864]

    # Create the plot
    plt.figure(figsize=(14, 8))

    # Plot for Classifier 1
    # plt.plot(epochs_1, training_loss_1, label='Training Loss (Classifier: 5 epoch, l.r. 5-e5)', marker='o')
    # plt.plot(epochs_1, validation_loss_1, label='Validation Loss (Classifier: 5 epoch, l.r. 5-e5)', marker='o')
    #
    # # Plot for Classifier 2
    # plt.plot(epochs_2, training_loss_2, label='Training Loss (Classifier: 10 epoch, l.r. 1-e5)', marker='o')
    # plt.plot(epochs_2, validation_loss_2, label='Validation Loss (Classifier: 10 epoch, l.r. 1-e5)', marker='o')

    # Plot for Classifier 3
    plt.plot(epochs_3, training_loss_3, label='Training Loss (Classifier: 19 epoch, l.r. 1-e5)', marker='o')
    plt.plot(epochs_3, validation_loss_3, label='Validation Loss (Classifier: 19 epoch, l.r. 1-e5)', marker='o')

    # Add titles and labels
    plt.title('Training and Validation Loss Over 19 epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Add grid
    # plt.grid(True)

    # Display the plot
    plt.show()
