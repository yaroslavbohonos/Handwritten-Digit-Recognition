# Handwritten Digit Recognition Neural Network с++

This C++ project implements a neural network for handwritten digit recognition. The network architecture consists of three layers with 785, 30, and 10 neurons respectively.

## Online Demo

An online demonstration is available where you can try out the trained model. Visit [Handwritten Digit Recognition Demo](https://handwritten-digit-recognition-demonstration.tiiny.site) to test the model with your own handwritten digits and see how well it performs.

## Image Preprocessing in web version

The process begins by scaling down the bounding box of the input drawing to a size of 20x20 pixels. The image is then centered on the center of mass of the pixels within a 28x28 image. This preprocessing step prepares the image for feeding it into the pre-trained neural network.

## Pre-Trained Model in web version

The neural network model used in this project is pre-trained using the MNIST dataset, following the same instructions as the MNIST training data. The model has learned to recognize handwritten digits based on this training.

## Usage

To use the neural network in your own C++ project, follow these steps:
1. Include the necessary files and dependencies in your project.
2. Unzip the Data.zip file, which contains the required training and testing data.
3. By running "recognition.cpp" you train the neural network (:
4. Use the pre-trained model "WeightsBiasesJSON.txt"

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to contribute, report issues, or make suggestions for improvements.
