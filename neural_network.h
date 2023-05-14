#pragma once

#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <random>
#include <array>
#include <vector>
#include <algorithm>

template <size_t inputs, size_t hidden_neurons, size_t output_neurons>
class NeuralNetwork
{
public:
    NeuralNetwork ()
    {
        /* Set the initial weights and biases to random numbers drawn from a 
        Gaussian distribution with a mean of 0 and standard deviation of 1.0 */
        std::random_device rd;
        std::mt19937 e2(rd());
        std::normal_distribution<float> dist(0, 1);
 
        for (float& f : m_hiddenLayerBiases)
            f = dist(e2);
 
        for (float& f : m_outputLayerBiases)
            f = dist(e2);
 
        for (float& f : m_hiddenLayerWeights)
            f = dist(e2);
 
        for (float& f : m_outputLayerWeights)
            f = dist(e2);
    }
 
    void Train (const MNISTData& trainingData, size_t miniBatchSize, float learningRate)
    {
        // Randomize the order of the training data to create mini-batches
        if (m_trainingOrder.size() != trainingData.NumImages())
        {
            m_trainingOrder.resize(trainingData.NumImages());
            size_t index = 0;
            for (size_t& v : m_trainingOrder)
            {
                v = index;
                ++index;
            }
        }
        static std::random_device rd;
        static std::mt19937 e2(rd());
        std::shuffle(m_trainingOrder.begin(), m_trainingOrder.end(), e2);
 
        // Process all minibatches until we are out of training examples
        size_t trainingIndex = 0;
        while (trainingIndex < trainingData.NumImages())
        {
            // Clear out minibatch derivatives. Sum up and then divide them at the end of the minibatch
            std::fill(m_miniBatchHiddenLayerBiasesDeltaCost.begin(), m_miniBatchHiddenLayerBiasesDeltaCost.end(), 0.0f);
            std::fill(m_miniBatchOutputLayerBiasesDeltaCost.begin(), m_miniBatchOutputLayerBiasesDeltaCost.end(), 0.0f);
            std::fill(m_miniBatchHiddenLayerWeightsDeltaCost.begin(), m_miniBatchHiddenLayerWeightsDeltaCost.end(), 0.0f);
            std::fill(m_miniBatchOutputLayerWeightsDeltaCost.begin(), m_miniBatchOutputLayerWeightsDeltaCost.end(), 0.0f);
 
            // Process the minibatch
            size_t miniBatchIndex = 0;
            while (miniBatchIndex < miniBatchSize && trainingIndex < trainingData.NumImages())
            {
                // Get the training item
                uint8_t imageLabel = 0;
                const float* pixels = trainingData.GetImage(m_trainingOrder[trainingIndex], imageLabel);
 
                // Run the forward pass of the network
                uint8_t labelDetected = ForwardPass(pixels, imageLabel);
 
                // Run the backward pass to get derivatives of the cost function
                BackwardPass(pixels, imageLabel);
 
                /* Adding current derivatives into the minibatch derivative arrays 
                we can average them at the end of the minibatch via division */
                for (size_t i = 0; i < m_hiddenLayerBiasesDeltaCost.size(); ++i)
                    m_miniBatchHiddenLayerBiasesDeltaCost[i] += m_hiddenLayerBiasesDeltaCost[i];
                for (size_t i = 0; i < m_outputLayerBiasesDeltaCost.size(); ++i)
                    m_miniBatchOutputLayerBiasesDeltaCost[i] += m_outputLayerBiasesDeltaCost[i];
                for (size_t i = 0; i < m_hiddenLayerWeightsDeltaCost.size(); ++i)
                    m_miniBatchHiddenLayerWeightsDeltaCost[i] += m_hiddenLayerWeightsDeltaCost[i];
                for (size_t i = 0; i < m_outputLayerWeightsDeltaCost.size(); ++i)
                    m_miniBatchOutputLayerWeightsDeltaCost[i] += m_outputLayerWeightsDeltaCost[i];
 
                // Add another item to the minibatch and used another training example
                ++trainingIndex;
                ++miniBatchIndex;
            }
 
            /* Divide the derivatives of the mini-series by the number of elements in 
            the mini-series to get the average value of the derivatives */
            float miniBatchLearningRate = learningRate / float(miniBatchIndex);

            /* Important: Instead of doing this explicitly like in the commented code below, 
            I did that implicitly above by dividing the learning rate by miniBatchIndex
            
            for (float& f : m_miniBatchHiddenLayerBiasesDeltaCost)  f /= float(miniBatchIndex);
            for (float& f : m_miniBatchOutputLayerBiasesDeltaCost)  f /= float(miniBatchIndex);
            for (float& f : m_miniBatchHiddenLayerWeightsDeltaCost) f /= float(miniBatchIndex);
            for (float& f : m_miniBatchOutputLayerWeightsDeltaCost) f /= float(miniBatchIndex); */
 
            // Application training to biases and weights
            for (size_t i = 0; i < m_hiddenLayerBiases.size(); ++i)
                m_hiddenLayerBiases[i] -= m_miniBatchHiddenLayerBiasesDeltaCost[i] * miniBatchLearningRate;
            for (size_t i = 0; i < m_outputLayerBiases.size(); ++i)
                m_outputLayerBiases[i] -= m_miniBatchOutputLayerBiasesDeltaCost[i] * miniBatchLearningRate;
            for (size_t i = 0; i < m_hiddenLayerWeights.size(); ++i)
                m_hiddenLayerWeights[i] -= m_miniBatchHiddenLayerWeightsDeltaCost[i] * miniBatchLearningRate;
            for (size_t i = 0; i < m_outputLayerWeights.size(); ++i)
                m_outputLayerWeights[i] -= m_miniBatchOutputLayerWeightsDeltaCost[i] * miniBatchLearningRate;
        }
    }
 
    // This function evaluates the network for the given input pixels and returns the predicted label, which can range from 0 to 9
    uint8_t ForwardPass (const float* pixels, uint8_t correctLabel)
    {
        for (size_t neuronIndex = 0; neuronIndex < hidden_neurons; ++neuronIndex)
        {
            float Z = m_hiddenLayerBiases[neuronIndex];
 
            for (size_t inputIndex = 0; inputIndex < inputs; ++inputIndex)
                Z += pixels[inputIndex] * m_hiddenLayerWeights[HiddenLayerWeightIndex(inputIndex, neuronIndex)];
 
            m_hiddenLayerOutputs[neuronIndex] = 1.0f / (1.0f + std::exp(-Z));
        }
 
        for (size_t neuronIndex = 0; neuronIndex < output_neurons; ++neuronIndex)
        {
            float Z = m_outputLayerBiases[neuronIndex];
 
            for (size_t inputIndex = 0; inputIndex < hidden_neurons; ++inputIndex)
                Z += m_hiddenLayerOutputs[inputIndex] * m_outputLayerWeights[OutputLayerWeightIndex(inputIndex, neuronIndex)];
 
            m_outputLayerOutputs[neuronIndex] = 1.0f / (1.0f + std::exp(-Z));
        }
 
        // Finding the maximum value of the output layer and return the index as the label
        float maxOutput = m_outputLayerOutputs[0];
        uint8_t maxLabel = 0;
        for (uint8_t neuronIndex = 1; neuronIndex < output_neurons; ++neuronIndex)
        {
            if (m_outputLayerOutputs[neuronIndex] > maxOutput)
            {
                maxOutput = m_outputLayerOutputs[neuronIndex];
                maxLabel = neuronIndex;
            }
        }
        return maxLabel;
    }
 
    // Functions to get weights / bias values. They are used to make the JSON file
    const std::array<float, hidden_neurons>& GetHiddenLayerBiases () const { return m_hiddenLayerBiases; }
    const std::array<float, output_neurons>& GetOutputLayerBiases () const { return m_outputLayerBiases; }
    const std::array<float, inputs * hidden_neurons>& GetHiddenLayerWeights () const { return m_hiddenLayerWeights; }
    const std::array<float, hidden_neurons * output_neurons>& GetOutputLayerWeights () const { return m_outputLayerWeights; }
 
private:
 
    static size_t HiddenLayerWeightIndex (size_t inputIndex, size_t hiddenLayerNeuronIndex)
    {
        return hiddenLayerNeuronIndex * inputs + inputIndex;
    }
 
    static size_t OutputLayerWeightIndex (size_t hiddenLayerNeuronIndex, size_t outputLayerNeuronIndex)
    {
        return outputLayerNeuronIndex * hidden_neurons + hiddenLayerNeuronIndex;
    }
 
    /* This function calculates the gradient needed for training by backpropagating the error of 
    the network, using the neuron output values from the forward pass. It determines the error 
    by comparing the label predicted by the network to the correct label */

    void BackwardPass (const float* pixels, uint8_t correctLabel)
    {
        // Since we are proceeding backwards, we are starting with the output layer
        for (size_t neuronIndex = 0; neuronIndex < output_neurons; ++neuronIndex)
        {
            float desiredOutput = (correctLabel == neuronIndex) ? 1.0f : 0.0f;
 
            float deltaCost_deltaO = m_outputLayerOutputs[neuronIndex] - desiredOutput;
            float deltaO_deltaZ = m_outputLayerOutputs[neuronIndex] * (1.0f - m_outputLayerOutputs[neuronIndex]);
 
            m_outputLayerBiasesDeltaCost[neuronIndex] = deltaCost_deltaO * deltaO_deltaZ;
 
            // Calculating deltaCost/deltaWeight for each weight going into the neuron
            for (size_t inputIndex = 0; inputIndex < hidden_neurons; ++inputIndex)
                m_outputLayerWeightsDeltaCost[OutputLayerWeightIndex(inputIndex, neuronIndex)] = m_outputLayerBiasesDeltaCost[neuronIndex] * m_hiddenLayerOutputs[inputIndex];
        }
 
        for (size_t neuronIndex = 0; neuronIndex < hidden_neurons; ++neuronIndex)
        {
            /* To calculate the error (deltaCost/deltaBias) for each hidden neuron we are following these steps:

            1. Multiply the deltaCost/deltaDestinationZ, which is already calculated and stored in 
            m_outputLayerBiasesDeltaCost[destinationNeuronIndex], by the weight connecting the source and target neurons. 
            This multiplication gives the error value for the neuron
            2. Multiply the neuron's output (O) by (1 - O) to obtain deltaO/deltaZ
            3. Compute deltaCost/deltaZ by multiplying the error by deltaO/deltaZ
            
            By following these steps, you can calculate the error (deltaCost/deltaBias) for each hidden neuron */

            float deltaCost_deltaO = 0.0f;
            for (size_t destinationNeuronIndex = 0; destinationNeuronIndex < output_neurons; ++destinationNeuronIndex)
                deltaCost_deltaO += m_outputLayerBiasesDeltaCost[destinationNeuronIndex] * m_outputLayerWeights[OutputLayerWeightIndex(neuronIndex, destinationNeuronIndex)];
            float deltaO_deltaZ = m_hiddenLayerOutputs[neuronIndex] * (1.0f - m_hiddenLayerOutputs[neuronIndex]);
            m_hiddenLayerBiasesDeltaCost[neuronIndex] = deltaCost_deltaO * deltaO_deltaZ;
 
            // Calculating deltaCost/deltaWeight for each weight going into the neuron
            for (size_t inputIndex = 0; inputIndex < inputs; ++inputIndex)
                m_hiddenLayerWeightsDeltaCost[HiddenLayerWeightIndex(inputIndex, neuronIndex)] = m_hiddenLayerBiasesDeltaCost[neuronIndex] * pixels[inputIndex];
        }
    }
 
private:
 
    // Weights and Biases 
    std::array<float, inputs * hidden_neurons>          m_hiddenLayerWeights;
    std::array<float, hidden_neurons * output_neurons>  m_outputLayerWeights;

    std::array<float, hidden_neurons>                   m_hiddenLayerBiases;
    std::array<float, output_neurons>                   m_outputLayerBiases;
    
    // Neuron activation values (known as "O" values)
    std::array<float, hidden_neurons>                   m_hiddenLayerOutputs;
    std::array<float, output_neurons>                   m_outputLayerOutputs;
 
    // Derivatives of biases and weights for a training example
    std::array<float, hidden_neurons>                   m_hiddenLayerBiasesDeltaCost;
    std::array<float, output_neurons>                   m_outputLayerBiasesDeltaCost;
 
    std::array<float, inputs * hidden_neurons>          m_hiddenLayerWeightsDeltaCost;
    std::array<float, hidden_neurons * output_neurons>  m_outputLayerWeightsDeltaCost;
 
    // Average of all items in minibatch (Derivatives of biases and weights for the minibatch)
    std::array<float, hidden_neurons>                   m_miniBatchHiddenLayerBiasesDeltaCost;
    std::array<float, output_neurons>                   m_miniBatchOutputLayerBiasesDeltaCost;
 
    std::array<float, inputs * hidden_neurons>          m_miniBatchHiddenLayerWeightsDeltaCost;
    std::array<float, hidden_neurons * output_neurons>  m_miniBatchOutputLayerWeightsDeltaCost;
 
    // Used for minibatch generation
    std::vector<size_t>                                 m_trainingOrder;
};