#define _CRT_SECURE_NO_WARNINGS
#include "timer.h"
#include "data_loader.h"
#include "neural_network.h"

// Setting to "1" shows error after each training and writes them in Error.csv file
// It may slow down the training process by 50%.
#define REPORT_ERROR_WHILE_TRAINING() 1
 
const size_t c_numInputNeurons = 785;
const size_t c_numHiddenNeurons = 30; // Setting up more neurons may give better result of neural network accuracy
const size_t c_numOutputNeurons = 10;

const size_t c_trainingEpochs = 30;
const size_t c_miniBatchSize = 10;
const float c_learningRate = 3.0f;

// The datasets used for training and testing a model
MNISTData g_trainingData;
MNISTData g_testData;
 
// neural network
NeuralNetwork <c_numInputNeurons, c_numHiddenNeurons, c_numOutputNeurons> g_neuralNetwork;
 
float GetDataAccuracy (const MNISTData& data)
{
    size_t correctItems = 0;
    for (size_t i = 0, c = data.NumImages(); i < c; ++i)
    {
        uint8_t label;
        const float* pixels = data.GetImage(i, label);
        uint8_t detectedLabel = g_neuralNetwork.ForwardPass(pixels, label);

        if (detectedLabel == label)
            ++correctItems;
    }
    return float(correctItems) / float(data.NumImages());
}
 
int main (int argc, char** argv)
{
    // Loading the MNIST data
    if (!g_trainingData.Load(true) || !g_testData.Load(false))
    {
        printf("Could not load the MNIST data!\n");
        return 1;
    }
 
    #if REPORT_ERROR_WHILE_TRAINING()
    FILE *file = fopen("Error.csv","w+t");
    if (!file)
    {
        printf("Could not open 'Error.csv' for writing!\n");
        return 2;
    }
    fprintf(file, "\"Training Data Accuracy\",\"Testing Data Accuracy\"\n");
    #endif

    {
        Timer timer("The training time:  ");
 
        // We report error before each training of neural network
        for (size_t epoch = 0; epoch < c_trainingEpochs; ++epoch)
        {
            #if REPORT_ERROR_WHILE_TRAINING()
                float accuracyTraining = GetDataAccuracy(g_trainingData);
                float accuracyTest = GetDataAccuracy(g_testData);
                printf("Training data accuracy: %0.2f%%\n", 100.0f*accuracyTraining);
                printf("Test data accuracy: %0.2f%%\n\n", 100.0f*accuracyTest);
                fprintf(file, "\"%f\",\"%f\"\n", accuracyTraining, accuracyTest);
            #endif
 
            printf("Training the epoch %zu / %zu...\n", epoch+1, c_trainingEpochs);
            g_neuralNetwork.Train(g_trainingData, c_miniBatchSize, c_learningRate);
            printf("\n");
        }
    }
 
    // report final error
    float accuracyTraining = GetDataAccuracy(g_trainingData);
    float accuracyTest = GetDataAccuracy(g_testData);
    printf("\nFinal training data accuracy: %0.2f%%\n", 100.0f*accuracyTraining);
    printf("Final test data accuracy: %0.2f%%\n\n", 100.0f*accuracyTest);
 
    #if REPORT_ERROR_WHILE_TRAINING()
        fprintf(file, "\"%f\",\"%f\"\n", accuracyTraining, accuracyTest);
        fclose(file);
    #endif
 
    // Write out the final weights and biases as JSON for use in the web demo
    {
        FILE* file = fopen("WeightsBiasesJSON.txt", "w+t");
        fprintf(file, "{\n");
 
        // network structure
        fprintf(file, "  \"InputNeurons\":%zu,\n", c_numInputNeurons);
        fprintf(file, "  \"HiddenNeurons\":%zu,\n", c_numHiddenNeurons);
        fprintf(file, "  \"OutputNeurons\":%zu,\n", c_numOutputNeurons);
 
        // HiddenBiases
        auto hiddenBiases = g_neuralNetwork.GetHiddenLayerBiases();
        fprintf(file, "  \"HiddenBiases\" : [\n");
        for (size_t i = 0; i < hiddenBiases.size(); ++i)
        {
            fprintf(file, "    %f", hiddenBiases[i]);
            if (i < hiddenBiases.size() -1)
                fprintf(file, ",");
            fprintf(file, "\n");
        }
        fprintf(file, "  ],\n");
 
        // HiddenWeights
        auto hiddenWeights = g_neuralNetwork.GetHiddenLayerWeights();
        fprintf(file, "  \"HiddenWeights\" : [\n");
        for (size_t i = 0; i < hiddenWeights.size(); ++i)
        {
            fprintf(file, "    %f", hiddenWeights[i]);
            if (i < hiddenWeights.size() - 1)
                fprintf(file, ",");
            fprintf(file, "\n");
        }
        fprintf(file, "  ],\n");
 
        // OutputBiases
        auto outputBiases = g_neuralNetwork.GetOutputLayerBiases();
        fprintf(file, "  \"OutputBiases\" : [\n");
        for (size_t i = 0; i < outputBiases.size(); ++i)
        {
            fprintf(file, "    %f", outputBiases[i]);
            if (i < outputBiases.size() - 1)
                fprintf(file, ",");
            fprintf(file, "\n");
        }
        fprintf(file, "  ],\n");
 
        // OutputWeights
        auto outputWeights = g_neuralNetwork.GetOutputLayerWeights();
        fprintf(file, "  \"OutputWeights\" : [\n");
        for (size_t i = 0; i < outputWeights.size(); ++i)
        {
            fprintf(file, "    %f", outputWeights[i]);
            if (i < outputWeights.size() - 1)
                fprintf(file, ",");
            fprintf(file, "\n");
        }
        fprintf(file, "  ]\n");
 
        // The end of training the neural network
        fprintf(file, "}\n");
        fclose(file);
    }
    
    return 0;
}
