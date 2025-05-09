# CNN Transfer Learning: CIFAR to MNIST

This project demonstrates how to apply transfer learning on a convolutional neural network (CNN) trained on the CIFAR dataset and refine it for classification tasks on the MNIST dataset. The solution leverages TensorFlow and Keras to build, train, and evaluate the model.

## Features

- **Transfer Learning**: Load a pre-trained model and refine it for a new classification task.
- **Data Preprocessing**: Resize and preprocess input data from MNIST to match the input requirements of the CIFAR model.
- **Model Freezing**: Retain pre-trained layers and fine-tune new layers for MNIST.
- **Evaluation**: Evaluate the accuracy of the refined model on test data.
