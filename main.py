# This is a backpropagation neural network with 3 layers (input, hidden, output)
# The input layer has 3 neurons, the hidden layer has 3 neurons and the output layer has 1 neuron
# All the neurons are fully connected
# We use the sigmoid function as the activation function
# We also use the batch gradient descent algorithm to train the network
# The problem we are trying to solve is the XOR problem, for 3 bits
# The weights are initialized randomly using a normal distribution with mean 0 and standard deviation 1

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Initialzing weihts and biases
# Using a normal distribution with mean 0 and standard deviation 1
def initialize_weights_and_biases():
    input_to_hidden_weights = np.random.normal(0, 1, (3, 3))
    hidden_to_output_weights = np.random.normal(0, 1, (3, 1))
    hidden_biases = np.random.normal(0, 1, (1, 3))
    output_biases = np.random.normal(0, 1, (1, 1))
    return input_to_hidden_weights, hidden_to_output_weights, hidden_biases, output_biases


def train(input_to_hidden_weights, hidden_to_output_weights, hidden_biases, output_biases, inputs, outputs, learning_rate=0.1, epochs=50000):

    # Training loop
    for epoch in range(epochs):

        # each iteration we folow minus the gradient towards the minmum, and update the weigts and biases
        # to caculate the gradient we use the chain rule
        # The backpropaation is really just a more eficient and faster way of using the chain rule to caculate the gradient

        # Forward pas (calculating all the outputs, since the ouput values are used when cacaulating the derivatives)
        # We use vectors to make caculations faster, so the inputs are a 2d array with 8 rows and 3 columns
        hidden_layer_input = np.dot(
            inputs, input_to_hidden_weights) + hidden_biases
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(
            hidden_layer_output, hidden_to_output_weights) + output_biases
        output_layer_output = sigmoid(output_layer_input)

        # Calculating error and gradients

        # the derivitve of a square of the error distance is 2 times the distance
        # we omit the 2 since it is a constant multiplied by the learning rate, and we can just change the learning rate
        output_error = outputs - output_layer_output
        output_delta = output_error * sigmoid_derivative(output_layer_input)

        hidden_error = np.dot(output_delta, hidden_to_output_weights.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_input)

        # Updating weights and biases
        hidden_to_output_weights += np.dot(hidden_layer_output.T,
                                           output_delta) * learning_rate
        output_biases += np.sum(output_delta, axis=0,
                                keepdims=True) * learning_rate
        input_to_hidden_weights += np.dot(inputs.T,
                                          hidden_delta) * learning_rate
        hidden_biases += np.sum(hidden_delta, axis=0,
                                keepdims=True) * learning_rate

        # Print loss every 1000 epochs
        if epoch % 1000 == 0:
            loss = np.mean(np.square(output_error))
            print(f"Epoch {epoch}, Loss: {loss}")

    return input_to_hidden_weights, hidden_to_output_weights, hidden_biases, output_biases, output_layer_output


# add main function

def main():

    # XOR data and expected outputs
    # 8 possible inputs since there are 8 possible combinations of 3 bits
    inputs = np.array([[0, 0, 0],
                       [0, 0, 1],
                       [0, 1, 0],
                       [0, 1, 1],
                       [1, 0, 0],
                       [1, 0, 1],
                       [1, 1, 0],
                       [1, 1, 1]])

    outputs = np.array([[0],
                        [1],
                        [1],
                        [0],
                        [1],
                        [0],
                        [0],
                        [1]])
    # Initializing weights and biases
    input_to_hidden_weights, hidden_to_output_weights, hidden_biases, output_biases = initialize_weights_and_biases()

    # Training the neural network
    input_to_hidden_weights, hidden_to_output_weights, hidden_biases, output_biases, output_layer_output = train(
        input_to_hidden_weights, hidden_to_output_weights, hidden_biases, output_biases, inputs, outputs)

    # Testing the trained neural network
    print("\nTrained neural network outputs:")
    print(output_layer_output)
    # Rounding the output to the nearest integer
    output_layer_output = np.round(output_layer_output)
    print("\nRounded neural network outputs:")
    print(output_layer_output)


if __name__ == "__main__":

    main()
