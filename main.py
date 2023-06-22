# This is a backpropagation neural network with 3 layers (input, hidden, output)
# The input layer has 3 neurons, the hidden layer has 3 neurons and the output layer has 1 neuron
# All the neurons are fully connected
# We use the sigmoid function as the activation function
# We also use the batch gradient descent algorithm to train the network
# The problem we are trying to solve is the XOR problem, for 3 bits
# The weights are initialized randomly using a normal distribution with mean 0 and standard deviation 1

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Initialzing weihts and biases
# Using a normal distribution with mean 0 and standard deviation 1
def initialize_weights_and_biases(number_of_hidden_neurons=3):
    input_to_hidden_weights = np.random.normal(
        0, 1, (3, number_of_hidden_neurons))
    hidden_to_output_weights = np.random.normal(
        0, 1, (number_of_hidden_neurons, 1))
    hidden_biases = np.random.normal(0, 1, (1, number_of_hidden_neurons))
    output_biases = np.random.normal(0, 1, (1, 1))
    return input_to_hidden_weights, hidden_to_output_weights, hidden_biases, output_biases


def train(input_to_hidden_weights, hidden_to_output_weights, hidden_biases, output_biases, inputs, outputs, learning_rate=0.1, epochs=2000):

    # initialize the loss array
    loss_array = np.zeros(2000)

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

        loss = np.mean(np.square(output_error))
        loss_array[epoch] = loss

    return input_to_hidden_weights, hidden_to_output_weights, hidden_biases, output_biases, output_layer_output, loss_array


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

    # run the training function a 100 times each time with different random weights and biases
    # Plot the average loss across all the runs, as a function of the number of epochs

    average_loss_array_3_neurons = np.zeros(2000)

    # run the training function 100 times
    for i in range(100):
        input_to_hidden_weights, hidden_to_output_weights, hidden_biases, output_biases = initialize_weights_and_biases()
        input_to_hidden_weights, hidden_to_output_weights, hidden_biases, output_biases, output_layer_output, loss_array = train(
            input_to_hidden_weights, hidden_to_output_weights, hidden_biases, output_biases, inputs, outputs)
        average_loss_array_3_neurons += loss_array

        # print the output of the network
        # first we round the output to 0 or 1
        output_layer_output = np.round(output_layer_output)
        print("the output of network number " + str(i + 1) + " is:")
        print(output_layer_output)

    average_loss_array_3_neurons /= 100

    # now we will try with 6 neurons in the hidden layer

    average_loss_array_6_neurons = np.zeros(2000)

    # run the training function 100 times
    for i in range(100):
        input_to_hidden_weights, hidden_to_output_weights, hidden_biases, output_biases = initialize_weights_and_biases(
            6)
        input_to_hidden_weights, hidden_to_output_weights, hidden_biases, output_biases, output_layer_output, loss_array = train(
            input_to_hidden_weights, hidden_to_output_weights, hidden_biases, output_biases, inputs, outputs)
        average_loss_array_6_neurons += loss_array

        # print the output of the network
        # first we round the output to 0 or 1
        output_layer_output = np.round(output_layer_output)
        print("the output of network number " + str(i + 1) + " is:")
        print(output_layer_output)

    average_loss_array_6_neurons /= 100

    # plot the average loss for both networks
    plt.plot(average_loss_array_3_neurons, label="3 neurons")
    plt.plot(average_loss_array_6_neurons, label="6 neurons")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    main()
